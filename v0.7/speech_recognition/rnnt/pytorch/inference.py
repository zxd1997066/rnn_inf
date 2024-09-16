# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from tqdm import tqdm
import toml
from dataset import AudioToTextDataLayer
from helpers import process_evaluation_batch, process_evaluation_epoch, add_blank_label, print_dict
from decoders import RNNTGreedyDecoder
from model_rnnt import RNNT
from preprocessing import AudioPreprocessing
import torch
import random
import numpy as np
import pickle
import time
import torchvision
import os

# import sys
# from IPython import embed
# def excepthook(type, value, traceback):
#     embed()
# sys.excepthook = excepthook


def parse_args():
    parser = argparse.ArgumentParser(description='Jasper')
    parser.add_argument("--batch_size", default=16,
                        type=int, help='data batch size')
    parser.add_argument("--ipex", action='store_true', default=False, help='use ipex')
    parser.add_argument('--precision', type=str, default="float32",
                        help='precision, float32, bfloat16')
    parser.add_argument("--jit", action='store_true', default=False, help='use jit script')
    parser.add_argument('--channels_last', type=int, default=1,
                        help='use channels last format')
    parser.add_argument("-t", "--profile", action='store_true',
                    help="Trigger profile on current topology.")
    parser.add_argument("--steps", default=None,
                        help='if not specified do evaluation on full dataset. otherwise only evaluates the specified number of iterations for each worker', type=int)
    parser.add_argument("--model_toml", type=str,
                        help='relative model configuration path given dataset folder')
    parser.add_argument("--dataset_dir", type=str,
                        help='absolute path to dataset folder')
    parser.add_argument("--val_manifest", type=str,
                        help='relative path to evaluation dataset manifest file')
    parser.add_argument("--ckpt", default=None, type=str,
                        required=True, help='path to model checkpoint')
    parser.add_argument("--pad_to", default=None, type=int,
                        help="default is pad to value as specified in model configurations. if -1 pad to maximum duration. If > 0 pad batch to next multiple of value")
    parser.add_argument("--cudnn_benchmark",
                        action='store_true', help="enable cudnn benchmark")
    parser.add_argument("--save_prediction", type=str, default=None,
                        help="if specified saves predictions in text form at this location")
    parser.add_argument("--logits_save_to", default=None,
                        type=str, help="if specified will save logits to path")
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--cuda",
                        action='store_true', help="use cuda", default=False)
    parser.add_argument("--compile", action='store_true', default=False,
                    help="enable torch.compile")
    parser.add_argument("--backend", type=str, default='inductor',
                    help="enable torch.compile backend")
    parser.add_argument("--triton_cpu", action='store_true', default=False,
                    help="enable triton_cpu")
    return parser.parse_args()


def eval(
        data_layer,
        audio_processor,
        encoderdecoder,
        greedy_decoder,
        labels,
        args):
    """performs inference / evaluation
    Args:
        data_layer: data layer object that holds data loader
        audio_processor: data processing module
        encoderdecoder: acoustic model
        greedy_decoder: greedy decoder
        labels: list of labels as output vocabulary
        args: script input arguments
    """
    logits_save_to = args.logits_save_to
    if args.triton_cpu:
        print("run with triton cpu backend")
        import torch._inductor.config
        torch._inductor.config.cpu_backend="triton"
    encoderdecoder.eval()
    with torch.no_grad():
        _global_var_dict = {
            'predictions': [],
            'transcripts': [],
            'logits': [],
        }

        total_time = 0
        total_seq_len = 0
        dry_run = 0
        batch_time_list = []
        for it, data in enumerate(tqdm(data_layer.data_iterator)):
            # jit
            if args.jit and it == 0:
                with torch.no_grad():
                    try:
                        encoderdecoder = torch.jit.trace(encoderdecoder, data, check_trace=False)
                        print("---- Use trace model.")
                    except:
                        encoderdecoder = torch.jit.script(encoderdecoder)
                        print("---- Use script model.")
                    if args.ipex:
                        encoderdecoder = torch.jit.freeze(encoderdecoder)
            fw_start_time = time.time()
            if args.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                    (t_audio_signal_e, t_a_sig_length_e,
                        transcript_list, t_transcript_e,
                        t_transcript_len_e) = audio_processor(data)
                profile_iter = args.steps if args.steps is not None else len(data_layer.data_iterator)
                if it == int(profile_iter/2):
                    import pathlib
                    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                    if not os.path.exists(timeline_dir):
                        os.makedirs(timeline_dir)
                    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                                "rnnt" + str(it) + '-' + str(os.getpid()) + '.json'
                    print(timeline_file)
                    prof.export_chrome_trace(timeline_file)
                    table_res = prof.key_averages().table(sort_by="cpu_time_total")
                    print(table_res)
                    # self.save_profile_result(timeline_dir + torch.backends.quantized.engine + "_result_average.xlsx", table_res)
            else:
                (t_audio_signal_e, t_a_sig_length_e,
                    transcript_list, t_transcript_e,
                    t_transcript_len_e) = audio_processor(data)

            # t_log_probs_e, (_, _) = torch.jit.trace(encoderdecoder,
            #     ((t_audio_signal_e, t_transcript_e),
            #      (t_a_sig_length_e, t_transcript_len_e),),
            # )

            # This is basically totaly useless. The encoder doesn't mean
            # anything by themsleves, in the case of RNN-T
            # t_log_probs_e, (_, _) = encoderdecoder(
            #     ((t_audio_signal_e, t_transcript_e),
            #      (t_a_sig_length_e, t_transcript_len_e),)
            # )
            t_predictions_e = greedy_decoder.decode(
                t_audio_signal_e, t_a_sig_length_e)
            fw_end_time = time.time()

            values_dict = dict(
                predictions=[t_predictions_e],
                transcript=transcript_list,
                transcript_length=t_transcript_len_e,
            )
            process_evaluation_batch(
                values_dict, _global_var_dict, labels=labels)

            if dry_run < 3:
                dry_run += 1
                continue

            total_time += fw_end_time - fw_start_time
            iter_seq_len = t_audio_signal_e.size()[0]*t_audio_signal_e.size()[1]
            total_seq_len += iter_seq_len
            print("Iteration: {}, inference time: {} sec, seq length: {}".format(it, fw_end_time - fw_start_time, iter_seq_len), flush=True)
            batch_time_list.append((fw_end_time - fw_start_time) / iter_seq_len * 1000)
            if args.steps is not None and it + 1 >= args.steps:
                break

        print("\n", "-"*20, "Summary", "-"*20)
        latency = total_time / total_seq_len * 1000
        throughput = total_seq_len / total_time
        print("inference latency:\t {:.3f} ms".format(latency))
        print("inference Throughput:\t {:.2f} seqs/s".format(throughput))
        # P50
        batch_time_list.sort()
        p50_latency = batch_time_list[int(len(batch_time_list) * 0.50) - 1]
        p90_latency = batch_time_list[int(len(batch_time_list) * 0.90) - 1]
        p99_latency = batch_time_list[int(len(batch_time_list) * 0.99) - 1]
        print('Latency P50:\t %.3f ms\nLatency P90:\t %.3f ms\nLatency P99:\t %.3f ms\n'\
                % (p50_latency, p90_latency, p99_latency))

        wer = process_evaluation_epoch(_global_var_dict)
        print("==========>>>>>>Evaluation WER: {0}\n".format(wer))
        if args.save_prediction is not None:
            with open(args.save_prediction, 'w') as fp:
                fp.write('\n'.join(_global_var_dict['predictions']))
        if logits_save_to is not None:
            logits = []
            with open(logits_save_to, 'wb') as f:
                pickle.dump(logits, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_profile_result(filename, table):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    keys = ["Name", "Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
            "CPU time avg", "Number of Calls"]
    for j in range(len(keys)):
        worksheet.write(0, j, keys[j])

    lines = table.split("\n")
    for i in range(3, len(lines)-4):
        words = lines[i].split(" ")
        j = 0
        for word in words:
            if not word == "":
                worksheet.write(i-2, j, word)
                j += 1
    workbook.close()


def main(args):
    random.seed(args.seed)
    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    print("CUDNN BENCHMARK ", args.cudnn_benchmark)
    if args.cuda:
        assert(torch.cuda.is_available())

    model_definition = toml.load(args.model_toml)
    dataset_vocab = model_definition['labels']['labels']
    ctc_vocab = add_blank_label(dataset_vocab)

    val_manifest = args.val_manifest
    featurizer_config = model_definition['input_eval']

    if args.pad_to is not None:
        featurizer_config['pad_to'] = args.pad_to if args.pad_to >= 0 else "max"

    print('model_config')
    print_dict(model_definition)
    print('feature_config')
    print_dict(featurizer_config)
    data_layer = None

    data_layer = AudioToTextDataLayer(
        dataset_dir=args.dataset_dir,
        featurizer_config=featurizer_config,
        manifest_filepath=val_manifest,
        labels=dataset_vocab,
        batch_size=args.batch_size,
        pad_to_max=featurizer_config['pad_to'] == "max",
        shuffle=False)
    audio_preprocessor = AudioPreprocessing(**featurizer_config)

    model = RNNT(
        feature_config=featurizer_config,
        rnnt=model_definition['rnnt'],
        num_classes=len(ctc_vocab)
    )

    if args.ckpt is not None:
        print("loading model from ", args.ckpt)
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    # model = torch.jit.script(model)

    audio_preprocessor.featurizer.normalize = "per_feature"

    if args.cuda:
        audio_preprocessor.cuda()
    if args.channels_last:
        audio_preprocessor = audio_preprocessor.to(memory_format=torch.channels_last)
        print("---- Use CL audio_preprocessor")
    audio_preprocessor.eval()
    if args.compile:
        audio_preprocessor = torch.compile(audio_preprocessor, backend=args.backend, options={"freezing": True})
    # IPEX
    if args.ipex:
        import intel_extension_for_pytorch as ipex
        print("Running with IPEX...")
        if args.precision == "bfloat16":
            audio_preprocessor = ipex.optimize(audio_preprocessor, dtype=torch.bfloat16, inplace=True)
        else:
            audio_preprocessor = ipex.optimize(audio_preprocessor, dtype=torch.float32, inplace=True)

    eval_transforms = []
    if args.cuda:
        eval_transforms.append(lambda xs: [x.cuda() for x in xs])
    eval_transforms.append(lambda xs: [*audio_preprocessor(xs[0:2]), *xs[2:]])
    # These are just some very confusing transposes, that's all.
    # BxFxT -> TxBxF
    eval_transforms.append(lambda xs: [xs[0].permute(2, 0, 1), *xs[1:]])
    eval_transforms = torchvision.transforms.Compose(eval_transforms)

    if args.cuda:
        model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("---- Use CL model")
    # ipex
    model.eval()
    if args.ipex:
        if args.precision == "bfloat16":
            model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
        else:
            model = ipex.optimize(model, dtype=torch.float32, inplace=True)

    # Ideally, I would jit this as well... But this is just the constructor...
    greedy_decoder = RNNTGreedyDecoder(len(ctc_vocab) - 1, model)
    if args.precision == "bfloat16":
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.bfloat16):
            eval(
                data_layer=data_layer,
                audio_processor=eval_transforms,
                encoderdecoder=model,
                greedy_decoder=greedy_decoder,
                labels=ctc_vocab,
                args=args)
    elif args.precision == "float16":
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.half):
            eval(
                data_layer=data_layer,
                audio_processor=eval_transforms,
                encoderdecoder=model,
                greedy_decoder=greedy_decoder,
                labels=ctc_vocab,
                args=args)
    else:
        eval(
            data_layer=data_layer,
            audio_processor=eval_transforms,
            encoderdecoder=model,
            greedy_decoder=greedy_decoder,
            labels=ctc_vocab,
            args=args)


if __name__ == "__main__":
    args = parse_args()

    print_dict(vars(args))

    main(args)
