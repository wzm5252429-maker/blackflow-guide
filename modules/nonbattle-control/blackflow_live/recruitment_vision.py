"""Ground ticket selection and emergency-hire previews from the current image.

This module never confirms an operator hire. A visible footer can establish an
occluding recruitment screen even when its cards are not readable yet.
"""
from dataclasses import dataclass
from hashlib import sha256
import math
import re

import numpy as np

from .models import ObservedAction


_TICKET_PROFESSIONS = {name + '招募券': name
                       for name in ('先锋', '近卫', '重装', '狙击', '术师', '医疗', '辅助', '特种')}


def _text(value):
    return re.sub(r'\s', '', value)


def _center(span):
    x,y,w,h = span.bbox
    return x+w/2,y+h/2


def _valid(span, width, height, confidence=.90):
    if not math.isfinite(span.confidence) or span.confidence < confidence:
        return False
    if len(span.bbox) != 4 or not all(math.isfinite(v) for v in span.bbox):
        return False
    x,y,w,h = span.bbox
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x+w <= width and y+h <= height


def _key(kind, text, bbox):
    return 'recruit:'+kind+':'+sha256((text+'|'+str(tuple(bbox))).encode()).hexdigest()[:14]


def _overlaps(a, b):
    ax,ay,aw,ah = a.bbox
    bx,by,bw,bh = b.bbox
    area = max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    return area > min(aw*ah,bw*bh)*.35


def _verified_ticket_title(image, span, ocr):
    name = _text(span.text)
    if name not in _TICKET_PROFESSIONS:
        return None
    if span.confidence >= .90:
        return name,span.confidence
    if not hasattr(ocr, 'recognize_crop'):
        return None
    # Recheck only the existing complete title, not a prefix or another card.
    import cv2
    x,y,w,h = span.bbox
    margin = max(2,round(h*.28))
    left,top = max(0,round(x)-margin),max(0,round(y)-margin)
    right,bottom = min(image.shape[1],round(x+w)+margin),min(image.shape[0],round(y+h)+margin)
    crop = image[top:bottom,left:right]
    if not crop.size:
        return None
    gray = cv2.cvtColor(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
    readings = [ocr.recognize_crop(pixels) for pixels in (crop,gray)]
    if all(_text(value)==name and math.isfinite(score) and score>=.90 for value,score in readings):
        return name,min(score for _,score in readings)
    return None


@dataclass(frozen=True)
class RecruitmentScreen:
    confidence: float
    actions: tuple[ObservedAction, ...]
    diagnostics: tuple[str, ...]
    evidence: dict


def _initial_tickets(image, spans, headers, ocr):
    height,width = image.shape[:2]
    if len(headers) != 1:
        return RecruitmentScreen(.94,(),('initial_recruitment_header_ambiguous',),{'kind':'initial_tickets'})
    buttons = sorted((span for span in spans if _valid(span,width,height)
                      and _text(span.text)=='招募' and .30*height<_center(span)[1]<.88*height),
                     key=lambda span: span.bbox)
    assigned = [[] for _ in buttons]
    for title in spans:
        if not _valid(title,width,height,.75) or not .20*height<_center(title)[1]<.70*height:
            continue
        if _text(title.text) not in _TICKET_PROFESSIONS:
            continue
        tx,ty = _center(title)
        candidates = []
        for index,button in enumerate(buttons):
            bx,by = _center(button)
            if (.06*height < by-ty < .40*height
                and abs(tx-bx) <= max(button.bbox[2]*1.5,width*.045)):
                candidates.append((abs(tx-bx),index))
        candidates.sort()
        if not candidates or (len(candidates)>1 and candidates[1][0]-candidates[0][0]<max(3,width*.008)):
            continue
        assigned[candidates[0][1]].append(title)
    actions = []
    for index,button in enumerate(buttons):
        # Overlapping controls and two possible titles remain ambiguous, even
        # if one of those OCR readings has a slightly higher confidence.
        if len(assigned[index]) != 1 or any(_overlaps(button,other) for other_index,other in enumerate(buttons) if other_index != index):
            continue
        title = assigned[index][0]
        verified = _verified_ticket_title(image,title,ocr)
        if verified is None:
            continue
        name,confidence = verified
        actions.append(ObservedAction(_key('ticket',name,button.bbox),name,'ui',button.bbox,
                                      min(button.confidence,confidence),metadata={
                                          'operation':'select_recruit_ticket','source':'ocr_initial_recruitment',
                                          'grounded':True,'preview_only':True,'selection_stage':'ticket_preview',
                                          'ticket_name':name,'ticket_profession':_TICKET_PROFESSIONS[name],
                                          'observed_text':button.text,'title_bbox':list(title.bbox),
                                      }))
    diagnostics = () if actions and len(actions)==len(buttons) else ('initial_recruitment_ticket_unreadable',)
    return RecruitmentScreen(min(.96,headers[0].confidence),tuple(actions),diagnostics,
                             {'kind':'initial_tickets','header_bbox':list(headers[0].bbox),
                              'visible_button_count':len(buttons)})


def detect_recruitment_screen(image: np.ndarray, spans, *, ocr, operators) -> RecruitmentScreen | None:
    """Use exact titles/footers; returned actions replace obscured background UI.

    ``operators`` maps complete normalized names to ``{'id': ..., 'name': ...}``.
    It must be a caller-verified identity catalog, not names inferred from this
    screenshot. Callers retain battle/ending priority before using this result.
    """
    height,width = image.shape[:2]
    spans = tuple(spans)
    headers = [span for span in spans if _valid(span,width,height,.94)
               and _text(span.text)=='初始招募' and .20*width<_center(span)[0]<.80*width
               and _center(span)[1]<.15*height]
    if headers:
        return _initial_tickets(image,spans,headers,ocr)
    # This is an occlusion cue, not a clickable hire. A weak full label must
    # not restore the map underneath; independent card evidence gates previews.
    hires = [span for span in spans if _valid(span,width,height,.75)
             and _text(span.text)=='雇佣' and .55*width<_center(span)[0]<.90*width
             and .65*height<_center(span)[1]<.88*height]
    footers = []
    for hire in hires:
        for leave in spans:
            if (not _valid(leave,width,height) or _text(leave.text)!='离开'
                or not .78*width<_center(leave)[0]<.99*width):
                continue
            if (abs(_center(hire)[1]-_center(leave)[1]) <= max(hire.bbox[3],leave.bbox[3])
                and .06*width<_center(leave)[0]-_center(hire)[0]<.35*width):
                footers.append((hire,leave))
    if not footers:
        return None
    if len(footers) != 1:
        return RecruitmentScreen(.94,(),('emergency_hire_footer_ambiguous',),{'kind':'emergency_hire'})
    hire,leave = footers[0]
    names = []
    for span in spans:
        if (not _valid(span,width,height) or not .42*width<_center(span)[0]<.94*width
            or not .12*height<_center(span)[1]<min(.73*height,hire.bbox[1]-.07*height)):
            continue
        name = _text(span.text)
        operator = operators.get(name)
        if (not operator or _text(operator.get('name',''))!=name
            or not isinstance(operator.get('id'),str) or not operator['id']):
            continue
        names.append((span,operator))
    names.sort(key=lambda value:value[0].bbox)
    names = [(span,operator) for index,(span,operator) in enumerate(names)
             if not any(_overlaps(span,other) for other_index,(other,_) in enumerate(names) if other_index != index)]
    diagnostics = ['emergency_hire_confirmation_requires_observation']
    evidence = {'kind':'emergency_hire','hire_bbox':list(hire.bbox),'leave_bbox':list(leave.bbox),
                'hire_label_confidence':hire.confidence,'reliable_operator_cards':len(names)}
    if hire.confidence < .90:
        separate_cards = any(
            operator['id'] != other_operator['id']
            and (abs(_center(span)[0]-_center(other)[0]) > width*.10
                 or abs(_center(span)[1]-_center(other)[1]) > height*.08)
            for index,(span,operator) in enumerate(names) for other,other_operator in names[index+1:])
        explicit_heading = any(_valid(span,width,height) and _text(span.text).startswith('应急雇佣')
                               and _center(span)[0] > width*.40 and _center(span)[1] < height*.20
                               for span in spans)
        if not separate_cards and not (explicit_heading and names):
            # The paired footer is enough to stop clicking through it, but not
            # enough to call an ambiguous card/menu target actionable.
            return RecruitmentScreen(min(.84,hire.confidence),(),
                                     ('emergency_hire_screen_evidence_ambiguous',),evidence)
        diagnostics.append('emergency_hire_footer_low_confidence')
    actions = []
    for index,(span,operator) in enumerate(names):
        actions.append(ObservedAction(_key('operator',span.text,span.bbox),span.text,'operator_preview',span.bbox,
                                      span.confidence,metadata={
                                          'operation':'event','source':'ocr_emergency_recruitment',
                                          'grounded':True,'preview_only':True,'selection_stage':'operator_preview',
                                          'operator_name':operator['name'],'operator_id':operator['id'],
                                          'observed_text':span.text,
                                      }))
    actions.append(ObservedAction(_key('leave',leave.text,leave.bbox),leave.text,'ui',leave.bbox,
                                  leave.confidence,metadata={
                                      'operation':'leave','source':'ocr_emergency_recruitment',
                                      'grounded':True,'ends_node':True,'observed_text':leave.text,
                                  }))
    if len(actions)==1:
        diagnostics.append('emergency_hire_operator_unrecognized')
    return RecruitmentScreen(min(.94,hire.confidence,leave.confidence),tuple(actions),tuple(diagnostics),
                             evidence)
