# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""

import json
import logging
import os

import torch
from ml_swissknife import utils
from transformers import HfArgumentParser, MODEL_WITH_LM_HEAD_MAPPING, set_seed
from transformers.models.gpt2 import GPT2Tokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from fastDP import PrivacyEngine
from E2E_utils.compiled_args import (DataTrainingArguments, ModelArguments, PrivacyArguments,
                            TrainingArguments)
from E2E_utils.misc import get_all_datasets, get_prompt_dataset
from E2E_utils.trainer import Trainer
from KFOptimizer import KFOptimizer
import tqdm

# from transformers.utils import logging as tlogging
logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, PrivacyArguments)
    )
    model_args, data_args, training_args, privacy_args = parser.parse_args_into_dataclasses()

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments
    privacy_args: PrivacyArguments

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use "
            f"--overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Debug mode
    if training_args.debug:
        import warnings
        warnings.filterwarnings("error")

    # Config.
    if 'gpt2' in model_args.model_name_or_path:
        from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel
        config = GPT2Config.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
        config.return_dict = True
        config.tie_word_embeddings = False

        # Tokenizer; `bos_token` and `eos_token` is the same for GPT2; both are 50256.
        tokenizer = GPT2Tokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

        # Model.
        gpt_model = GPT2LMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )
        print(f'base gpt2 model: {model_args.model_name_or_path}')
        print(gpt_model)
    elif 'gptj' in model_args.model_name_or_path:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        gpt_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        config = gpt_model.config
    elif 'min' in model_args.model_name_or_path:
        from mingpt.model import GPT
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=model_args.cache_dir)
        minGPT_config = GPT.get_default_config()
        minGPT_config.vocab_size = len(tokenizer)
        minGPT_config.block_size = tokenizer.model_max_length
        # print(minGPT_config)
        # print(GPT.__dict__)
        minGPT_config.model_type = None
        minGPT_config.n_layer = 2
        minGPT_config.n_head = 8
        minGPT_config.n_embd = 768
        gpt_model = GPT(minGPT_config)
        config = gpt_model.config


    # Clone the embedding into the lm_head for better initialization.
    # if not 'min' in model_args.model_name_or_path:
    lm_head = gpt_model.get_output_embeddings()
    embedding = gpt_model.get_input_embeddings()
    lm_head.weight.data.copy_(embedding.weight.data)
    print(f'Cloning initial embedding into lm_head, '
        f'checking norms... \n'
        f'\tlm_head: {lm_head.weight.norm()}, embedding: {embedding.weight.norm()}')
    torch.testing.assert_allclose(lm_head.weight, embedding.weight)
    del lm_head, embedding

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.model_max_length
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Adjust tokenizer and model embeddings.
    print('adapt tokenizer to include [PAD]')
    print(f'before len(tokenizer) = {len(tokenizer)}')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print(f'after len(tokenizer) = {len(tokenizer)}')
    print('tokenizer.eos_token:', tokenizer.eos_token, tokenizer.eos_token_id)
    print('tokenizer.bos_token:', tokenizer.bos_token, tokenizer.bos_token_id)

    print('adapt the size of lm_head and input_embeddings to include [PAD]')
    print('use avg-based initialization')

    input_embeddings_before = gpt_model.get_input_embeddings().weight
    lm_head_before = gpt_model.get_output_embeddings().weight
    gpt_model.resize_token_embeddings(len(tokenizer))

    input_embeddings_after = gpt_model.get_input_embeddings().weight
    lm_head_after = gpt_model.get_output_embeddings().weight
    print(
        f'before lm_head.weight.size() = {lm_head_before.size()}, '
        f'input_embeddings_before.size() = {input_embeddings_before.size()}'
    )
    print(
        f'after lm_head.weight.size() = {lm_head_after.size()}, '
        f'after input_embeddings_after.size() = {input_embeddings_after.size()}'
    )
    torch.testing.assert_allclose(lm_head_before, lm_head_after[:-1])
    print('pre-chunk equal for lm_head')
    torch.testing.assert_allclose(input_embeddings_before, input_embeddings_after[:-1])
    print('pre-chunk equal for input_embeddings')
    lm_head_after.data[-1] = lm_head_before.mean(dim=0)
    input_embeddings_after.data[-1] = input_embeddings_before.mean(dim=0)

    print('double check: ')
    print('embedding size', gpt_model.get_input_embeddings().weight.size())
    print('lm_head size', gpt_model.get_output_embeddings().weight.size())
    model = gpt_model

    train_dataset, val_dataset, eval_dataset, data_collator = get_all_datasets(
        config=config,
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        model_args=model_args,
    )

    # Materialize the prompts.
    generation_stuff = dict(
        train_prompts=get_prompt_dataset(file_path=data_args.train_prompt_file, tokenizer=tokenizer),
        val_prompts=get_prompt_dataset(file_path=data_args.val_prompt_file, tokenizer=tokenizer),
        eval_prompts=get_prompt_dataset(file_path=data_args.eval_prompt_file, tokenizer=tokenizer),
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        model_args=model_args,
        data_args=data_args,
        privacy_args=privacy_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        generation_stuff=generation_stuff,
    )

    # Massage the parameters.
    if model_args.attention_only:
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            if 'c_attn.weight' in name:
                param.requires_grad_(True)
    elif model_args.bias_only:
        for name, param in model.named_parameters():
            if '.bias' not in name:
                param.requires_grad_(False)        
        if model_args.static_lm_head and hasattr(model, 'lm_head'):
            model.lm_head.requires_grad_(False)
    else:
        model.requires_grad_(True)
        if model_args.static_lm_head:
            model.get_output_embeddings().requires_grad_(False)
        if model_args.static_embedding:
            model.get_input_embeddings().requires_grad_(False)
            model.transformer.wpe.requires_grad_(False)
    print(f"bias_only: {model_args.bias_only} | attention_only: {model_args.attention_only}")
        
        
    params = tuple(param for param in model.parameters() if param.requires_grad)
    names = tuple(name for name, param in model.named_parameters() if param.requires_grad)
    num_trainable_params = sum(param.numel() for param in params)
    print(f"Number of trainable params: {num_trainable_params / 1e6:.4f} million")
    print(f'Number of total params: {sum(param.numel() for param in model.parameters()) / 1e6:.3f} million')

    print(json.dumps(names, indent=4))

    # TODO: Using a single gigantic parameter group is okay only when `weight_decay` is 0.
    #   Biases and LM parameters should not be decayed perhaps even with privacy.
    optimizer = torch.optim.AdamW(
        params=params,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )
    trainer.optimizer = optimizer

    # Create the lr_scheduler.
    try:
        num_GPUs=torch.distributed.get_world_size()
    except:
        num_GPUs=1
        
    if training_args.logical_batch_size!=None:
        trainer.args.gradient_accumulation_steps=training_args.logical_batch_size/training_args.per_device_train_batch_size/num_GPUs
    else:
        training_args.logical_batch_size=trainer.args.gradient_accumulation_steps*training_args.per_device_train_batch_size*num_GPUs

    num_update_steps_per_epoch = len(trainer.get_train_dataloader()) // trainer.args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    t_total = int(num_update_steps_per_epoch * trainer.args.num_train_epochs)
    if training_args.lr_decay:
        trainer.lr_scheduler = get_linear_schedule_with_warmup(
            trainer.optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=t_total,
        )
    else:
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.)

    if training_args.kf:
        optimizer = KFOptimizer(params, optimizer,training_args.kappa, training_args.gamma)
    trainer.optimizer = optimizer

    # Hacky way to set noise_multiplier.
    if privacy_args.non_private:
        privacy_args.noise_multiplier = 0.
        privacy_args.per_example_max_grad_norm = None
    else:
        origin_params=None if model_args.bias_only or model_args.attention_only or training_args.deepspeed_config else ['wte','wpe']

        privacy_engine = PrivacyEngine(
            module=model,
            batch_size=training_args.logical_batch_size,
            sample_size=len(train_dataset),
            epochs=training_args.num_train_epochs,
            max_grad_norm=privacy_args.per_example_max_grad_norm,
            noise_multiplier=privacy_args.noise_multiplier,
            target_epsilon=privacy_args.target_epsilon,
            target_delta=privacy_args.target_delta,
            accounting_mode=privacy_args.accounting_mode,
            clipping_mode=privacy_args.clipping_mode,
            clipping_fn=privacy_args.clipping_fn,
            clipping_style=privacy_args.clipping_style,
            origin_params=origin_params,
            num_GPUs=num_GPUs,
            torch_seed_is_fixed=True,
        )

        # Originally, these could have been null.
        privacy_args.noise_multiplier = privacy_engine.noise_multiplier
        privacy_args.target_delta = privacy_engine.target_delta

        print('privacy_args: ')
        print(json.dumps(privacy_args.__dict__, indent=4))
        if not training_args.deepspeed_config:
            privacy_engine.attach(optimizer)
        # privacy_engine.attach(optimizer)

    # Training.
    if training_args.do_train:
        all_args = {
            **training_args.__dict__,
            **data_args.__dict__,
            **model_args.__dict__,
            **privacy_args.__dict__,
        }
        utils.jdump(
            all_args,
            os.path.join(training_args.output_dir, 'argparse.json'),
            default=lambda x: str(x),
        )

        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

        logger.info("*** Train ***")
        logger.info(
            f"Training set size: {len(train_dataset)}, "
            f"per_device_train_batch_size: {training_args.per_device_train_batch_size}, "
            f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}"
        )
        trainer.train(model_path=None)
        if training_args.save_at_last:
            trainer.save_model()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        output = trainer.evaluate(log_results=False)
        utils.jdump(
            output,
            os.path.join(training_args.output_dir, "final_results.json"),
        )

        logger.info("***** Eval results *****")
        logger.info(output)


if __name__ == "__main__":
    main()
