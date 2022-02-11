nohup parlai train_model -t blended_skill_talk:concepts,wizard_of_wikipedia:concepts,convai2:normalized_concepts,empathetic_dialogues:concepts \
--multitask-weights 1,3,3,3 -veps 0.25 --attention-dropout 0.0 --batchsize 128 --model transformer/generator --embedding-size 2560 --ffn-size 10240 \
--variant prelayernorm --n-heads 32 --n-positions 128 --n-encoder-layers 2 --n-decoder-layers 24 --history-add-global-end-token end --delimiter '  ' \
--dict-tokenizer bytelevelbpe  --dropout 0.1 --fp16 True --init-model zoo:blender/reddit_3B/model --dict-file zoo:blender/reddit_3B/model.dict \
--label-truncate 128 --log_every_n_secs 10 -lr 7e-06 --lr-scheduler reduceonplateau --lr-scheduler-patience 3 --optimizer adam --relu-dropout 0.0 \
--activation gelu --model-parallel true --save-after-valid True --text-truncate 128 --truncate 128 --warmup_updates 100 --fp16-impl mem_efficient \
--update-freq 2 --gradient-clip 0.1 --skip-generation True -vp 10 -vmt ppl -vmm min --model-file models/concepts/test_train_27B --verbose \
--wblog True --wandb-entity ilyalas --wandb-project all_data_27B --wandb-name concepts &