"""Read regular recruitment cards from one supplied screenshot, without input.

The hope bar animates a prospective deduction after selection. Its displayed
value is therefore *not* an available balance for a paid confirmation. The
currently supported confirmation is an explicitly observed zero-cost card.
"""
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import math
import re

import numpy as np

from .models import ObservedAction


@dataclass(frozen=True)
class RecruitmentCardsScreen:
    confidence: float
    actions: tuple[ObservedAction, ...]
    resources: dict
    diagnostics: tuple[str, ...]
    evidence: dict


@dataclass(frozen=True)
class _CardTitle:
    text: str
    confidence: float
    bbox: tuple


def _text(value):
    return re.sub(r'[\s“”"‘’·•:：，,。.!！?？<>/\\]', '', value)


def _valid(span, image, confidence=.90):
    height,width = image.shape[:2]
    if not math.isfinite(span.confidence) or span.confidence < confidence:
        return False
    if len(span.bbox) != 4 or not all(math.isfinite(v) for v in span.bbox):
        return False
    x,y,w,h = span.bbox
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x+w <= width and y+h <= height


def _center(box):
    x,y,w,h = box
    return x+w/2,y+h/2


def _inside(inner, outer):
    x,y,w,h = inner
    ox,oy,ow,oh = outer
    return x >= ox and y >= oy and x+w <= ox+ow and y+h <= oy+oh


def _crop(image, box):
    x,y,w,h = box
    left,top = max(0,round(x)),max(0,round(y))
    right,bottom = min(image.shape[1],round(x+w)),min(image.shape[0],round(y+h))
    return image[top:bottom,left:right]


@lru_cache(maxsize=3)
def _template(name):
    import cv2
    path = Path(__file__).with_name('assets') / (name+'.png')
    return cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)


def _yellow(image):
    a = image.astype(np.int16)
    return ((a[:,:,1] > a[:,:,0]*1.2) & (a[:,:,2] > 90) & (a[:,:,1] > 100)).astype(np.uint8)*255


def _blue_fraction(image):
    if not image.size:
        return 0.
    a = image.astype(np.int16)
    return float(((a[:,:,0]-a[:,:,2] > 25) & (a[:,:,1]-a[:,:,2] > 20)
                  & (a[:,:,1] > 50)).mean())


def _patch_hit(image, name, scale, *, yellow=False, threshold=.90):
    """Match within a caller-owned visual region, never across another card."""
    import cv2
    template = _template(name)
    size = (max(3,round(template.shape[1]*scale)),max(3,round(template.shape[0]*scale)))
    if image.shape[1] < size[0] or image.shape[0] < size[1]:
        return None
    template = cv2.resize(template,size,interpolation=cv2.INTER_LINEAR)
    if yellow:
        target,template = _yellow(image),_yellow(template)
    else:
        target = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    scores = cv2.matchTemplate(target,template,cv2.TM_CCOEFF_NORMED)
    _,score,_,(x,y) = cv2.minMaxLoc(scores)
    if not math.isfinite(score) or score < threshold:
        return None
    return (x,y,*size),float(score)


def _operator(span, image, ocr, operators):
    name = _text(span.text)
    operator = operators.get(name)
    if (not operator or _text(operator.get('name','')) != name
        or not isinstance(operator.get('id'),str) or not operator['id'].startswith('char_')):
        return None
    if span.confidence >= .90:
        return operator,span.confidence
    if not hasattr(ocr,'recognize_crop'):
        return None
    # Re-read the complete observed title. A catalog prefix is never completed.
    import cv2
    crop = _crop(image,span.bbox)
    if not crop.size:
        return None
    readings = [ocr.recognize_crop(crop),
                ocr.recognize_crop(cv2.cvtColor(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR))]
    verified = [(text,score) for text,score in readings if math.isfinite(score) and score >= .90]
    if verified and all(_text(text) == name for text,_ in verified):
        return operator,min(score for _,score in verified)
    return None


def _cost(image, box, ocr, spans=()):
    """Read complete white price glyphs next to a verified hope icon."""
    if not hasattr(ocr,'recognize_crop'):
        return None
    import cv2
    crop = _crop(image,box)
    if not crop.size:
        return None
    prepared = cv2.cvtColor(255-((crop.min(axis=2)>80)*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    readings = [ocr.recognize_crop(crop),ocr.recognize_crop(prepared)]
    x,y,w,h = box
    reading_region = (x-h*.15,y-h*.1,w+h*.3,h*1.2)
    readings.extend((span.text,span.confidence) for span in spans
                    if _valid(span,image) and _inside(span.bbox,reading_region))
    accepted = set()
    for text,confidence in readings:
        text = text.strip().translate(str.maketrans({'O':'0','o':'0','Ｏ':'0'}))
        if math.isfinite(confidence) and confidence >= .90 and re.fullmatch(r'\d{1,2}',text):
            accepted.add(int(text))
    return next(iter(accepted)) if len(accepted)==1 else None


def _full_card_title(image, box, ocr, operators):
    """Read the whole title band independently of any incomplete global OCR."""
    if not hasattr(ocr,'recognize_crop'):
        return None
    import cv2
    x,y,w,h = box
    region = (x+w*.58,y+h*.73,w*.40,h*.25)
    crop = _crop(image,region)
    if not crop.size:
        return None
    readings = [ocr.recognize_crop(crop),ocr.recognize_crop(
        cv2.cvtColor(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR))]
    if not all(math.isfinite(score) and score >= .93 for _,score in readings):
        return None
    names = {_text(text) for text,_ in readings}
    if len(names)!=1:
        return None
    name = next(iter(names))
    operator = operators.get(name)
    if (not operator or _text(operator.get('name',''))!=name
        or not isinstance(operator.get('id'),str) or not operator['id'].startswith('char_')):
        return None
    confidence = min(score for _,score in readings)
    return _CardTitle(name,confidence,region),(operator,confidence)


def _card_boxes(image, footer):
    """Find complete dark card contours; cropped edge cards remain unavailable."""
    import cv2
    height,width = image.shape[:2]
    fx,fy,fw,fh = footer.bbox
    scale = fh/24
    mask = (image.max(axis=2)<130).astype(np.uint8)*255
    mask[:max(0,round(fh*3))] = 0
    mask[max(0,round(fy-fh*1.5)):] = 0
    mask[:,:round(width*.25)] = 0
    k = max(3,round(7*scale))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((k,k),np.uint8))
    _,_,stats,_ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for x,y,w,h,area in stats[1:]:
        if (2.4 < w/h < 3.6 and fh*3 < h < fh*6 and area > w*h*.58
            and x > width*.25 and x+w < width-2*scale):
            boxes.append(tuple(int(v) for v in (x,y,w,h)))
    return sorted(boxes,key=lambda b:(b[1],b[0]))


def _displayed_hope(image, scale):
    """Recognize the complete current zero cell, excluding the maximum cell.

    The low-resolution zero is confidently misread as nine by generic OCR.
    This small, provenance-recorded image template establishes only zero; other
    digits and unseen styles remain unknown instead of being guessed.
    """
    height,width = image.shape[:2]
    top = image[:round(height*.15)]
    hit = _patch_hit(top,'recruitment_hope_header',scale,threshold=.94)
    if hit is None:
        return None,{}
    (x,y,w,h),score = hit
    # Ratios are relative to the visually located header, not the frame edge.
    rect = (x+w*(89/76),y+h*(14/37),w*(24/76),h*(21/37))
    cell = _crop(image,rect)
    zero = _patch_hit(cell,'recruitment_hope_zero',scale,threshold=.965)
    evidence = {'header_bbox':[x,y,w,h],'header_confidence':score,
                'current_cell_bbox':list(rect),'reading_source':'current_zero_template'}
    if zero is None:
        return None,evidence
    evidence['zero_confidence'] = zero[1]
    return 0,evidence


def _action(label, kind, bbox, confidence, metadata):
    key = 'recruit-card:'+sha256((label+'|'+str(bbox)+'|'+kind).encode()).hexdigest()[:16]
    return ObservedAction(key,label,kind,bbox,confidence,metadata={
        'source':'regular_recruitment_cards','grounded':True,**metadata})


def detect_recruitment_cards(image: np.ndarray, spans, *, ocr, operators) -> RecruitmentCardsScreen | None:
    """Return occluding card UI, grounded actions and independently read facts.

    Callers give battle/ending priority, replace obscured background actions,
    and stamp each action with the same observation's ``source_frame_id``.
    ``operators`` is an exact verified name/id catalog, not an OCR guess list.
    """
    height,width = image.shape[:2]
    spans = tuple(spans)
    confirms = [s for s in spans if _valid(s,image,.75) and _text(s.text)=='确认招募'
                and _center(s.bbox)[0] > width*.65 and _center(s.bbox)[1] > height*.82]
    pairs = [(confirm,abandon) for confirm in confirms for abandon in spans
             if _valid(abandon,image) and _text(abandon.text)=='放弃'
             and abs(_center(abandon.bbox)[1]-_center(confirm.bbox)[1]) < max(abandon.bbox[3],confirm.bbox[3])
             and width*.03 < _center(confirm.bbox)[0]-_center(abandon.bbox)[0] < width*.30]
    if not pairs:
        return None
    evidence = {'kind':'regular_operator_cards'}
    if len(pairs)!=1:
        return RecruitmentCardsScreen(.94,(),{},('recruitment_footer_ambiguous',),evidence)
    confirm,abandon = pairs[0]
    evidence.update(confirm_bbox=list(confirm.bbox),abandon_bbox=list(abandon.bbox))
    scale = confirm.bbox[3]/24
    displayed,hope_evidence = _displayed_hope(image,scale)
    evidence['hope_display'] = {**hope_evidence,'displayed_hope':displayed,
                                'available_balance_verified':False}
    details = [(s,_operator(s,image,ocr,operators)) for s in spans
               if _valid(s,image,.75) and _center(s.bbox)[0] < width*.25
               and height*.07 < _center(s.bbox)[1] < height*.19]
    details = [(s,identity) for s,identity in details if identity is not None]
    selected_detail = details[0] if len(details)==1 else None
    placeholders = [s for s in spans if _valid(s,image,.75) and _center(s.bbox)[0] < width*.25
                    and '轻触右侧干员' in _text(s.text)]
    if placeholders:
        selected_detail = None
    evidence['detail_operator_id'] = selected_detail[1][0]['id'] if selected_detail else None
    cards = []
    for box in _card_boxes(image,confirm):
        x,y,w,h = box
        names = []
        name_region = (x+w*.53,y+h*.70,w*.46,h*.30)
        for span in spans:
            if _valid(span,image,.75) and _inside(span.bbox,name_region):
                identity = _operator(span,image,ocr,operators)
                if identity is not None:
                    names.append((span,identity))
        # Multiple identities on one visual card are ambiguous, even if OCR
        # happens to score one of them more highly.
        name = names[0] if len(names)==1 else None
        if not names:
            name = _full_card_title(image,box,ocr,operators)
        region_box = (x+w*.54,y+h*.44,w*.28,h*.28)
        region = _crop(image,region_box)
        icon = _patch_hit(region,'recruitment_hope_cost',scale,yellow=True,threshold=.68)
        icon_shape = _patch_hit(region,'recruitment_hope_cost',scale,threshold=.90)
        if (icon is None or icon_shape is None
            or abs(icon[0][0]-icon_shape[0][0]) > scale
            or abs(icon[0][1]-icon_shape[0][1]) > scale):
            icon = None
        cost,icon_box = None,None
        if icon is not None:
            (ix,iy,iw,ih),_ = icon
            ix += round(region_box[0]); iy += round(region_box[1])
            icon_box = (ix,iy,iw,ih)
            price_box = (ix+iw*1.40,iy,iw,ih*1.05)
            if _inside(price_box,box):
                cost = _cost(image,price_box,ocr,spans)
        blue = _blue_fraction(_crop(image,(x,y+h*.65,w,h*.30)))
        cards.append({'bbox':box,'name':name,'cost':cost,'icon_bbox':icon_box,'highlighted':blue>.25,
                      'lower_card_blue_fraction':blue})
    evidence['cards'] = [{'bbox':list(c['bbox']),'operator_id':c['name'][1][0]['id'] if c['name'] else None,
                           'hope_cost':c['cost'],'hope_icon_bbox':c['icon_bbox'],
                           'highlighted':c['highlighted'],'lower_card_blue_fraction':c['lower_card_blue_fraction']}
                          for c in cards]
    highlighted = [c for c in cards if c['highlighted']]
    selected = None
    if selected_detail and len(highlighted)==1 and highlighted[0]['name']:
        card = highlighted[0]
        if card['name'][1][0]['id'] == selected_detail[1][0]['id']:
            selected = card
    actions = []
    for card in cards:
        if not card['name'] or card is selected:
            continue
        span,(operator,confidence) = card['name']
        actions.append(_action(operator['name'],'operator_preview',span.bbox,confidence,{
            'operation':'event','preview_only':True,'selection_stage':'operator_preview',
            'operator_id':operator['id'],'operator_name':operator['name'],'card_bbox':list(card['bbox']),
            'observed_text':span.text}))
    diagnostics,resources = [],{}
    if not cards:
        diagnostics.append('recruitment_card_contours_unreadable')
    button_enabled = _valid(confirm,image) and _blue_fraction(_crop(image,confirm.bbox)) > .30
    evidence['button_enabled_observed'] = bool(button_enabled)
    if selected is None:
        diagnostics.append('recruitment_selection_not_verified')
    elif selected['cost'] is None:
        diagnostics.append('recruitment_selected_cost_unreadable')
    elif selected['cost'] > 0:
        diagnostics.append('recruitment_hope_display_may_be_post_selection')
    elif displayed is None:
        diagnostics.append('recruitment_hope_display_unreadable')
    elif not button_enabled:
        diagnostics.append('recruitment_confirm_not_enabled')
    else:
        # With a proven cost of zero, prospective and current balance coincide.
        resources['hope'] = displayed
        evidence['hope_display']['available_balance_verified'] = True
        span,(operator,confidence) = selected['name']
        actions.append(_action(confirm.text,'ui',confirm.bbox,min(confidence,confirm.confidence,selected_detail[1][1]),{
            'operation':'recruit_reserve','selection_stage':'operator_confirm','preview_only':False,
            'operator_id':operator['id'],'operator_name':operator['name'],'selected_operator_id':operator['id'],
            'button_enabled_observed':True,'hope_cost':0,'resource_costs':{'hope':0},
            'card_bbox':list(selected['bbox']),'detail_bbox':list(selected_detail[0].bbox),
            'hope_icon_bbox':list(selected['icon_bbox']),'observed_text':confirm.text}))
    if selected_detail and len(highlighted)==1 and highlighted[0]['cost'] not in (None,0):
        if 'recruitment_hope_display_may_be_post_selection' not in diagnostics:
            diagnostics.append('recruitment_hope_display_may_be_post_selection')
    return RecruitmentCardsScreen(min(.96,confirm.confidence,abandon.confidence),tuple(actions),resources,
                                  tuple(diagnostics),evidence)
