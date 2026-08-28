# Copyright 2022 David Scripka. All rights reserved.
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

# Imports
import pyaudio
import numpy as np
from openwakeword.model import Model
import argparse

# Parse input arguments
parser=argparse.ArgumentParser()
parser.add_argument(
    "--chunk_size",
    help="How much audio (in number of samples) to read from the microphone at once. "
         "Defaults to --step_samples. Reading larger chunks than the prediction step throws "
         "away the latency benefit of a smaller step, because results only arrive once per read.",
    type=int,
    default=None,
    required=False
)
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    default="",
    required=False
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite'",
    type=str,
    default='tflite',
    required=False
)
parser.add_argument(
    "--step_samples",
    help="Samples of audio per prediction; must divide 1280 evenly. 1280 = 80 ms (default), "
         "640 = 40 ms. Halving it cuts ~30 ms of detection latency for 2x the CPU.",
    type=int,
    default=1280,
    required=False
)
parser.add_argument(
    "--ncpu",
    help="CPU threads for the melspectrogram and embedding models. More threads lower the "
         "wall-clock cost of a prediction but raise total CPU; 2 is the useful maximum.",
    type=int,
    default=1,
    required=False
)

args=parser.parse_args()

# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size if args.chunk_size is not None else args.step_samples
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Load pre-trained openwakeword models.
#
# On a 4-core Raspberry Pi, `--step_samples 640 --ncpu 2` is the tuning that was measured:
# the 40 ms step cuts median detection latency by ~30 ms, and two threads leave headroom for
# the rest of the system. Both default to the library's conservative settings so that a
# single-core or power-constrained target is not silently charged for them.
oww_kwargs = dict(
    inference_framework=args.inference_framework,
    step_samples=args.step_samples,
    ncpu=args.ncpu,
)
if args.model_path != "":
    owwModel = Model(wakeword_models=[args.model_path], **oww_kwargs)
else:
    owwModel = Model(**oww_kwargs)

n_models = len(owwModel.models.keys())

# Run capture loop continuosly, checking for wakewords
if __name__ == "__main__":
    # Generate output string header
    print("\n\n")
    print("#"*100)
    print("Listening for wakewords...")
    print("#"*100)
    print("\n"*(n_models*3))

    while True:
        # Get audio
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Feed to openWakeWord model
        prediction = owwModel.predict(audio)

        # Column titles
        n_spaces = 16
        output_string_header = """
            Model Name         | Score | Wakeword Status
            --------------------------------------
            """

        for mdl in owwModel.prediction_buffer.keys():
            # Add scores in formatted table
            scores = list(owwModel.prediction_buffer[mdl])
            curr_score = format(scores[-1], '.20f').replace("-", "")

            output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {"--"+" "*20 if scores[-1] <= 0.5 else "Wakeword Detected!"}
            """

        # Print results table
        print("\033[F"*(4*n_models+1))
        print(output_string_header, "                             ", end='\r')
