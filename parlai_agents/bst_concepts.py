### copy to blended_skill_talk/agents.py

import parlai.utils.logging as logging
from parlai.utils.misc import str_to_msg
TOKEN_KNOWLEDGE = '__knowledge__'
TOKEN_END_KNOWLEDGE = '__endknowledge__'
class ConceptsTeacher(BlendedSkillTalkTeacher):
    def _setup_data(self, path):
        logging.info(f"Loading ParlAI text data: {path}")
        self.episodes = []
        self.num_exs = 0
        eps = []
        with PathManager.open(path, newline='\n', encoding='utf-8') as read:
            for line_no, line in enumerate(read, 1):
                msg = str_to_msg(line.rstrip('\n'))
                if msg and 'eval_labels' in msg:
                    raise ValueError(
                        f"It looks like you've written eval_labels as a key in your "
                        f"data file. This is not appropriate; labels will be converted "
                        f"for you automatically. This is happening on Line {line_no} "
                        f"in {path}. The line is:\n\t{line}"
                    )
                if msg and 'text' not in msg:
                    raise ValueError(
                        f'ParlaiDialogTeacher requires a "text" field in every '
                        f'entry, but one is missing in Line {line_no} in {path}. '
                        f'The line is:\n\t{line}'
                    )
                if msg and 'labels' not in msg:
                    raise ValueError(
                        f'ParlaiDialogTeacher requires a "labels" field in every '
                        f'entry, but one is missing in Line {line_no} in {path}. '
                        f'The line is:\n\t{line}'
                    )
                if msg and 'concepts' not in msg:
                    raise ValueError(
                        f'BlendedSkillTalkConceptsTeacher requires a "concepts" field in every '
                        f'entry, but one is missing in Line {line_no} in {path}. '
                        f'The line is:\n\t{line}'
                    )
                if msg:
                    self.num_exs += 1
                    concepts = msg['concepts'].replace("|",".")
                    text = msg['text'] + f'\n{TOKEN_KNOWLEDGE}{concepts}{TOKEN_END_KNOWLEDGE}'
                    msg.force_set('text',text)
                    del msg['concepts']
                    eps.append(msg)
                    if msg.get('episode_done', False):
                        self.episodes.append(eps)
                        eps = []
        if len(eps) > 0:
            # add last episode
            eps[-1].force_set('episode_done', True)
            self.episodes.append(eps)
        if len(self.episodes) == 1 and line_no > 100:
            logging.error(
                f'The data in {path} looks like one very long episode. If this '
                f'is intentional, you may ignore this, but you MAY have a bug in '
                f'your data.'
            )
