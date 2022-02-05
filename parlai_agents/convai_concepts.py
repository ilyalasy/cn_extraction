### copy to convai2/agents.py

class SelfConceptsTeacher(SelfOriginalTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except Exception:
            use_cands = True
        opt['datafile'] = _path(opt, 'self_original_concepts', use_cands)
        super().__init__(opt, shared)
