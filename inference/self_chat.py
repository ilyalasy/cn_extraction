### replace SelfChatWorld in blended_skill_talk/worlds.py
from parlai.tasks.blended_skill_talk.extract import extract_from_msg
TOKEN_KNOWLEDGE = '__knowledge__'
TOKEN_END_KNOWLEDGE = '__endknowledge__'
class SelfChatWorld(SelfChatBaseWorld):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        parser = parser.add_argument_group('BST SelfChat World')
        parser.add_argument(
            '--include-personas',
            type='bool',
            default=True,
            help='Include personas as input context, or not',
        )
        parser.add_argument(
            '--include-initial-utterances',
            type='bool',
            default=True,
            help='Include context conversation at beginning or not',
        )
        parser.add_argument(
            '--include-concepts',
            type='bool',
            default=False,
            help='Retrieve concepts from utterance or not',
        )
        return parser

    def init_contexts(self, shared=None):
        self.contexts_data = get_contexts_data(self.opt, shared=shared)

    def get_contexts(self):
        random.seed()
        p = random.choice(self.contexts_data)
        return [p[0], p[1]]

    def share(self):
        shared_data = super().share()
        shared_data['contexts_data'] = self.contexts_data
        return shared_data
    
    def _add_knowledge_to_act(self, act):
        text = act['text']
        concepts = extract_from_msg(text)
        knowledge = ' '.join(concepts)
        text += f'\n{TOKEN_KNOWLEDGE}{knowledge}{TOKEN_END_KNOWLEDGE}'
        act.force_set('text',text)
        return act

    def parley(self):
        if self.episode_done():
            self._end_episode()

        if self.turn_cnt == 0:
            self.acts = [None, None]
            # get any context for the beginning of the conversation
            self.contexts = self.get_contexts()

        self.seed_utterances = self._get_seed_utt_acts(self.episode_cnt, self.agents)

        if self.contexts:
            assert len(self.contexts) == 2
            # initial context
            for i in range(0, 2):
                context = Message(
                    {'text': self.contexts[i], 'episode_done': False, 'id': 'context'}
                )
                self.acts[i] = context
                self.agents[i].observe(validate(context))
            # clear contexts so they are only added once per episode
            self.contexts = None
        elif self.seed_utterances:
            # pop the next two seed messages (there may be less or more than 2 total)
            utts = self.seed_utterances[:2]
            self.seed_utterances = self.seed_utterances[2:]
            # process the turn
            for i in [0, 1]:
                # if we have a seed utterance, add it to the conversation
                if len(utts) > i:
                    self.acts[i] = utts[i]
                    if hasattr(self.agents[i], 'self_observe'):
                        self.agents[i].observe({'episode_done': False})
                        self.agents[i].self_observe(self.acts[i])
                else:
                    self.acts[i] = self.agents[i].act()
                self.agents[1 - i].observe(validate(self.acts[i]))
        else:
            # do regular loop
            acts = self.acts
            agents = self.agents

            acts[0] = agents[0].act()            
            acts[0] = self._add_knowledge_to_act(acts[0])
            agents[1].observe(validate(acts[0]))

            acts[1] = agents[1].act()
            acts[1] = self._add_knowledge_to_act(acts[1])
            agents[0].observe(validate(acts[1]))

        self.update_counters()
        self.turn_cnt += 1