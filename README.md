# Open-domain Chatbot Augmented with Commonsense Knowledge

Master Thesis work published on [the Conference "Lithuanian MSc Research in Informatics and ICT"](https://www.journals.vu.lt/open-series/article/view/27518/26857).

## Abstract
*Building an open-domain dialog system is a challenging task in current research.
In order to successfully maintain a conversation with human, a dialog system must develop many qualities: being engaging, empathetic, show a unique personality and having general knowledge about the world.
Prior research has shown that it is possible to develop such chat-bot system that combines these features, but this work explores this problem further. Most state-of-the-art dialogue systems are guided by unstructured knowledge such as Wikipedia articles, but there is a lack of research on how structured knowledge bases can be used for open-domain dialogue generation.
This work proposes usage of structured knowledge base ConceptNet for knowledge-grounded dialogue generation. Novel knowledge extraction algorithm is developed which is then used to incorporate knowledge into existing dialogue datasets.
Current state-of-the-art model BlenderBot is finetuned on newly created datasets and it is shown that knowledge augmentation of the dataset improved BlenderBot in terms of various automated metrics and according to human evaluation.*

## Small technical description

Baseline model, [BlenderBot 1](https://parl.ai/projects/recipes/), was fine-tuned on a knowledge-augmented datasets. Each original dataset ([BST](https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/blended_skill_talk), [ConvAI2](https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/convai2), [WoW](https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/wizard_of_wikipedia), [ED](https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/empathetic_dialogues)) was preprocessed by [knowledge extraction algorithm](extraction.py).
Developed algorithm extracts knowledge triples *(assertions)* from [ConceptNet](https://conceptnet.io/) and adds the most relevant ones to the inputted utterance. Relevance is described by cosine similarity between the utterance sentence embedding and the knowledge triple embedding (treated as a small sentence).
Extracted knowledge were appended to dataset messages, each ConceptNet relation was treated as a *special token*. The latest version of the algorithm also extracts knowledge from the *whole context* of the dialogue and not only the last utterance.

## Automated metrics
<img width="609" alt="image" src="https://user-images.githubusercontent.com/31886723/212345266-d71e5b92-0e27-400f-8680-3d2e912a0818.png">

## Human evaluation

There was an attempt to evaluate the developed model in a fashion similar to [ACUTE-EVAL](https://arxiv.org/abs/1909.03087). 
Although there were not enough resources to perform a full-scale crowdsourced survey, a small amount (~30) of friends and relatives were able to take a survey. 
One can still take the [survey](https://ilyalas6394.crowdsignal.net/chatbot-evaluation-1) if interested.

<img width="606" alt="image" src="https://user-images.githubusercontent.com/31886723/212350420-ae0eaef9-38f8-44a1-a8f1-b3a8f721659e.png">
