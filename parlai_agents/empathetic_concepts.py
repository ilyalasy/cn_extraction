### copy to empathetic_dialogues/agents.py

TOKEN_KNOWLEDGE = '__knowledge__'
TOKEN_END_KNOWLEDGE = '__endknowledge__'
class EmpatheticConceptsTeacher(EmpatheticDialoguesTeacher):
    def __init__(self, opt, shared=None):
        split = self.datatype.split(':')
        base_datatype = split[0] + '_concepts'
        self.datatype = ':'.join([base_datatype] + split[1:])
        super().__init__(opt, shared)
        
    def _setup_data(self, base_datatype):

        if self.opt.get('deepmoji') is not None:
            self.embed = np.load(self.opt['deepmoji'] + base_datatype + ".npy")

        if self.opt.get('fasttextloc') is not None and self.opt.get('prepend', -1) > 0:
            try:
                import fastText
            except ImportError:
                raise ImportError("Please run 'pip install fasttext'.")
            ftpath = self.opt['fasttextloc']
            ftmodel = fastText.FastText.load_model(ftpath)

        with PathManager.open(self.datapath) as f:
            df = f.readlines()

        turn_idx = 1
        responder_text_dialogue = []
        experiencer_text_dialogue = []
        self.data = []
        for i in range(1, len(df)):

            cparts = df[i - 1].strip().split(",")
            sparts = df[i].strip().split(",")

            if cparts[0] == sparts[0]:

                # Check that the turn number has incremented correctly
                turn_idx += 1
                assert (
                    int(cparts[1]) + 1 == int(sparts[1]) and int(sparts[1]) == turn_idx
                )

                contextt = cparts[5].replace("_comma_", ",")
                concepts = cparts[-1].replace(' ')
                contextt += f'\n{TOKEN_KNOWLEDGE}{concepts}{TOKEN_END_KNOWLEDGE}'
                label = sparts[5].replace("_comma_", ",")
                prompt = sparts[2]
                sit = sparts[3].replace("_comma_", ",")
                if len(sparts) == 10:
                    if sparts[8] != '':
                        inline_label_candidates = [
                            cand.replace("_comma_", ",").replace("_pipe_", "|")
                            for cand in sparts[8].split('|')
                        ]
                    else:
                        inline_label_candidates = []
                elif len(sparts) == 9:
                    inline_label_candidates = []
                else:
                    raise ValueError(f'Line {i:d} has the wrong number of fields!')

                context_emb, cand_emb = None, None
                if self.opt.get('deepmoji') is not None:
                    context_emb = self.embed[i - 2]
                    cand_emb = self.embed[i - 1]

                ft_ctx, ft_cand = None, None
                if (
                    self.opt.get('fasttextloc') is not None
                    and self.opt.get('prepend', -1) > 0
                ):
                    ft_ctx = ""
                    gettop, _ = ftmodel.predict(contextt, k=self.opt['prepend'])
                    for f in gettop:
                        ft_ctx = f.split("_")[-1] + " " + ft_ctx
                    ft_cand = ""
                    gettop, _ = ftmodel.predict(label, k=self.opt['prepend'])
                    for f in gettop:
                        ft_cand = f.split("_")[-1] + " " + ft_cand

                # Check if either the text or label are marked as being political
                is_political = '<POLITICAL>' in cparts[7] or '<POLITICAL>' in sparts[7]

                dialogue_parts = [
                    contextt,
                    label,
                    prompt,
                    sit,
                    context_emb,
                    cand_emb,
                    ft_ctx,
                    ft_cand,
                    inline_label_candidates,
                    is_political,
                ]

                if int(sparts[1]) % 2 == 0:
                    # experiencer is the "text" and responder is the "label"
                    experiencer_text_dialogue.append(dialogue_parts)
                else:
                    # responder is the "text" and experiencer is the "label"
                    responder_text_dialogue.append(dialogue_parts)

            else:

                # We've finished the previous episode, so add it to the data
                turn_idx = 1
                self.data += self._select_dialogues_to_add(
                    experiencer_text_dialogue, responder_text_dialogue
                )
                experiencer_text_dialogue = []
                responder_text_dialogue = []

        # Add in the final episode
        self.data += self._select_dialogues_to_add(
            experiencer_text_dialogue, responder_text_dialogue
        )
