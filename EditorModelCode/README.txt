Data preprocessing:
The AmericanPoliticians dataset must be augmented with edit labels by running
Levenshtein_word_distance.ipynb.

Transformer model:
run_seq2seq.py is a modified version of HuggingFace's run_translation.py,
modified to work with pre-edit and post-edit text from the same language
(English in this case).  For the original file, see 
https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py

Input to transformer model:
Before running this model, one must change the data to jsonlines format and append the
appropirate personalization using Make_jsonl.ipynb.
For the generator phase, one must additionally add another column to the data file,
called edit_string_predicted, which contains the predicted outputs from
the Annotate model (also see Make_jsonl.ipynb).

Running run_seq2seq.py:
This can be used for running both stages of the Annotate-Generate model, as follows:

    # ANNOTATE MODEL
    python run_seq2seq.py \
        --model_name_or_path t5-base \
        --do_train \
        --do_eval \
        --do_predict \
        --source_lang en \
        --target_lang en \
        --source_prefix "predict encyclopedia edits: " \
        --output_dir [OUTPUT_DIR] \
        --num_train_epochs=15 \
        --per_device_train_batch_size=3 \
        --per_device_eval_batch_size=3 \
        --save_strategy=epoch \
        --evaluation_strategy=epoch \
        --load_best_model_at_end True \
        --predict_with_generate \
        --train_file [TRAIN JSONL FILE] \
        --validation_file [VAL JSONL FILE] \
        --test_file [TRAIN+VAL+TEST JSON FILE]

    # GENERATE MODEL
    python run_seq2seq.py \
        --model_name_or_path t5-base \
        --do_train \
        --do_eval \
        --do_predict \
        --source_lang en \
        --target_lang en \
        --source_prefix "edit encyclopedia article: " \
        --output_dir [OUTPUT_DIR] \
        --num_train_epochs=15 \
        --per_device_train_batch_size=2 \
        --per_device_eval_batch_size=2 \
        --save_strategy=epoch \
        --evaluation_strategy=epoch \
        --load_best_model_at_end True \
        --predict_with_generate \
        --train_file [TRAIN JSONL FILE] \
        --validation_file [VAL JSONL FILE] \
        --test_file [TEST JSONL FILE]
