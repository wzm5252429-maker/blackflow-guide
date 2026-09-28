"""Ground a visible shop confirmation without reading through its background."""
from dataclasses import dataclass
import math
import re

import numpy as np

from .models import ObservedAction


def _text(value):
    return re.sub(r'[\s:："“”‘’]', '', value)


def _center(span):
    x,y,w,h = span.bbox
    return x+w/2,y+h/2


def _crop(image, box):
    x,y,w,h = box
    mx,my = max(2, round(h*.25)),max(2, round(h*.23))
    left,top = max(0, round(x)-mx),max(0, round(y)-my)
    right,bottom = min(image.shape[1], round(x+w)+mx),min(image.shape[0], round(y+h)+my)
    return image[top:bottom, left:right]


def _gray(crop, *, contrast=False):
    import cv2
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if contrast:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _button_confidence(image, span, ocr):
    if span.confidence >= .90:
        return span.confidence
    if span.confidence < .75 or not hasattr(ocr, 'recognize_crop'):
        return None
    crop = _crop(image, span.bbox)
    if not crop.size:
        return None
    # Recheck the existing complete label locally; different/partial readings
    # never get promoted by a neighboring title or the button's color.
    readings = [ocr.recognize_crop(_gray(crop, contrast=contrast)) for contrast in (False, True)]
    if all(_text(text) == _text(span.text) and math.isfinite(score) and score >= .90
           for text,score in readings):
        return min(score for _,score in readings)
    return None


@dataclass(frozen=True)
class ShopDialog:
    confidence: float
    actions: tuple[ObservedAction, ...]
    diagnostics: tuple[str, ...]
    evidence: dict

    def grounded_actions(self, resources):
        result = []
        for action in self.actions:
            if action.metadata.get('operation') == 'purchase':
                gold = resources.get('gold')
                if (not isinstance(gold, (int, float)) or isinstance(gold, bool)
                    or not math.isfinite(gold) or gold < action.metadata['price']):
                    continue
            result.append(action)
        return result


def _question_fact(image, rows, operation, item_names, ocr):
    if not rows or not hasattr(ocr, 'recognize_crop'):
        return None
    left = min(span.bbox[0] for span in rows)
    top = min(span.bbox[1] for span in rows)
    right = max(span.bbox[0]+span.bbox[2] for span in rows)
    bottom = max(span.bbox[1]+span.bbox[3] for span in rows)
    crop = _crop(image, (left,top,right-left,bottom-top))
    if not crop.size:
        return None
    pattern = (r'是否以(?:源石锭|[△▲Δ])?(\d{1,3})(?:源石锭)?的价格出售(.+?)[?？]?'
               if operation == 'sell' else
               r'是否(?:消耗|花费)(?:源石锭|[△▲Δ])?(\d{1,3})(?:源石锭)?购买(.+?)[?？]?')
    facts = []
    for pixels in (crop, _gray(crop)):
        value,confidence = ocr.recognize_crop(pixels)
        if not math.isfinite(confidence) or confidence < .90:
            continue
        match = re.fullmatch(pattern, _text(value))
        if match and match[2] in item_names:
            facts.append((int(match[1]), item_names[match[2]], confidence, value))
    if not facts or len({(price,name) for price,name,_,_ in facts}) != 1:
        return None
    return min(facts, key=lambda item: item[2])


def detect_shop_dialog(image: np.ndarray, spans, *, ocr, item_names) -> ShopDialog | None:
    """A verified confirm/cancel footer blocks interaction with the background.

    Catalog entries identify a fully read item only. All amounts come from a
    fresh OCR crop of the visible question, never from a shelf or catalog price.
    A trade additionally requires its complete question and unique item title;
    an incomplete dialog exposes only its verified cancel button.
    """
    height,width = image.shape[:2]
    confirmations = [span for span in spans if _text(span.text) in {'确认购买', '确认出售'}
                     and math.isfinite(span.confidence) and span.confidence >= .75
                     and width*.30 < _center(span)[0] < width*.95
                     and height*.25 < _center(span)[1] < height*.85]
    if not confirmations:
        return None
    footer_cancels = {}
    for candidate in confirmations:
        associated = [span for span in spans if _text(span.text) == '算了'
                      and math.isfinite(span.confidence) and span.confidence >= .90
                      and abs(_center(span)[1]-_center(candidate)[1]) <= max(candidate.bbox[3], span.bbox[3])
                      and candidate.bbox[3]*2 < _center(candidate)[0]-_center(span)[0] < width*.45]
        for span in associated:
            footer_cancels[id(span)] = span
    if not footer_cancels:
        return None
    if len(confirmations) != 1 or len(footer_cancels) != 1:
        # A transition can show both trade labels, or duplicate cancellation
        # targets. The footer still occludes the shelf/map, but proves no trade.
        actions = ()
        if len(footer_cancels) == 1:
            cancel = next(iter(footer_cancels.values()))
            actions = (ObservedAction('shop:cancel_confirmation', cancel.text, 'ui', cancel.bbox,
                                      cancel.confidence, metadata={
                                          'operation': 'event_advance', 'source': 'ocr_shop_dialog',
                                          'grounded': True, 'dialog_kind': 'ambiguous',
                                          'observed_text': cancel.text,
                                      }),)
        evidence = {'kind': 'ambiguous',
                    'confirm_bboxes': sorted(list(span.bbox) for span in confirmations),
                    'cancel_bboxes': sorted(list(span.bbox) for span in footer_cancels.values())}
        return ShopDialog(min(.95, *(span.confidence for span in footer_cancels.values())),
                          actions, ('shop_confirmation_footer_ambiguous',), evidence)
    confirm = confirmations[0]
    bx,by,bw,bh = confirm.bbox
    operation = 'sell' if _text(confirm.text) == '确认出售' else 'purchase'
    cancel = next(iter(footer_cancels.values()))
    button_confidence = _button_confidence(image, confirm, ocr)
    cancel_action = ObservedAction('shop:cancel_confirmation', cancel.text, 'ui', cancel.bbox,
                                  cancel.confidence, metadata={
                                      'operation': 'event_advance', 'source': 'ocr_shop_dialog',
                                      'grounded': True, 'dialog_kind': operation,
                                      'observed_text': cancel.text,
                                  })
    evidence = {'kind': operation, 'confirm_bbox': list(confirm.bbox),
                'cancel_bbox': list(cancel.bbox), 'question_bbox': None,
                'visible_titles': []}
    left,right = max(0, cancel.bbox[0]-width*.16),min(width, bx+bw+width*.10)
    question_prefix = '是否以' if operation == 'sell' else ('是否消耗', '是否花费')
    question_rows = [span for span in spans if span.confidence >= .85
                     and _text(span.text).startswith(question_prefix)
                     and left <= span.bbox[0] < right
                     and by-max(height*.16, bh*5) < _center(span)[1] < by-bh*.45]
    if len(question_rows) != 1:
        # Missing/weak question OCR does not make an already verified footer
        # disappear. Returning None here would re-enable the obscured map/shelf.
        if button_confidence is None:
            return None
        return ShopDialog(min(cancel.confidence, button_confidence, .95), (cancel_action,),
                          ('shop_confirmation_question_missing_or_ambiguous',), evidence)
    question = question_rows[0]
    rows = sorted((span for span in spans if span.confidence >= .85
                   and left <= span.bbox[0] and span.bbox[0]+span.bbox[2] <= right
                   and abs(_center(span)[1]-_center(question)[1]) <= max(3, question.bbox[3]*.6)),
                  key=lambda span: span.bbox[0])
    # A second currency or unrelated row is not borrowed to make a price.
    if any(b.bbox[0]-(a.bbox[0]+a.bbox[2]) > width*.05 for a,b in zip(rows, rows[1:])):
        rows = []
    titles = {item_names[_text(span.text)] for span in spans if span.confidence >= .90
              and _text(span.text) in item_names and left <= _center(span)[0] <= right
              and by-height*.50 <= _center(span)[1] < question.bbox[1]-bh*1.5}
    actions = [cancel_action]
    diagnostics = []
    evidence.update(question_bbox=list(question.bbox), visible_titles=sorted(titles))
    fact = _question_fact(image, rows, operation, item_names, ocr)
    if button_confidence is not None and fact is not None and titles == {fact[1]}:
        price,name,question_confidence,observed_question = fact
        actions.insert(0, ObservedAction('shop:'+operation+'_confirmation', confirm.text, 'ui', confirm.bbox,
                                        min(button_confidence, question_confidence), metadata={
                                            'operation': operation, 'source': 'ocr_shop_dialog',
                                            'grounded': True, 'item_name': name, 'price': price,
                                            'resource_delta': {'gold': price if operation == 'sell' else -price},
                                            'dialog_kind': operation, 'selection_stage': 'shop_confirmation',
                                            'observed_question': observed_question,
                                        }))
        evidence.update(item_name=name, price=price, observed_question=observed_question)
    else:
        diagnostics.append('shop_confirmation_incomplete_or_conflicting')
    return ShopDialog(min(cancel.confidence, question.confidence, .95), tuple(actions),
                      tuple(diagnostics), evidence)
