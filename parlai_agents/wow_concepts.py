### copy to wizard_of_wikipedia/agents.py

class ConceptsTeacher(BasicdialogTeacher):
    def get(self, episode_idx, entry_idx=0):
        d = self.data[episode_idx]
        episode_done = entry_idx == (self.len_episode(episode_idx) - 1)

        idx = entry_idx * 2
        first_speaker = d['dialog'][0]['speaker'].lower()
        if self.speaker_label != 'both' and self.speaker_label in first_speaker:
            idx += 1

        dialog_entry_1 = d['dialog'][idx]
        dialog_entry_2 = d['dialog'][idx + 1]

        text = dialog_entry_1['text']
        concepts = dialog_entry_1.get('concepts',[])
        concepts = '.'.join(concepts)        
        text += f'\n{TOKEN_KNOWLEDGE}{concepts}{TOKEN_END_KNOWLEDGE}'
        labels = [dialog_entry_2['text']]

        assert isinstance(self.add_topic, bool)
        if self.add_topic and entry_idx == 0:
            text = d.get('chosen_topic', '') + '\n' + text

        action = Message(
            {
                'id': 'ConceptsBasicDialog',
                'text': text,
                'labels': labels,
                'episode_done': episode_done,
            }
        )
        if 'label_candidates' in d:
            action['label_candidates'] = d['label_candidates']

        if self.speaker_label == 'wizard':
            action['chosen_topic'] = d.get('chosen_topic', '')

        return action
