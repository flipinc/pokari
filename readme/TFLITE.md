# TFLite
**TFLite support has ended since v0.1.0. All related codes will be removed gradually.**

## Why not TFLite?
// TODO
- tensorflow serving (you just upload your model once)
- increasing engineering (unicode points, unsupported operations by tflite, expertise required)

## Getting Started
Following (un)installations are required for tflite conversion after starting Tensorflow's Dockerfile:
```shell
pip uninstall warprnnt_tensorflow
pip install tensorflow==2.3.2 tensorflow-io==0.16 tensorflow-text==2.3
```

## TFLite Conversion
- As of 2021/2/17, tensorflow 2.4 does not work well with tflite. If you see errors such as 
`tensorflow.python.framework.errors_impl.InvalidArgumentError: Attempting to add a duplicate function with name`,
it is highly likely that reverting back tensorflow version might solve some issues (or at least give you directions to solve them). It is confirmed that following tensorflow version works.
    - tensorflow 2.3.2
- tf.string is not supported in TFLite, so all models outputs Unicode instead.
- Make sure the converter version matches with the runtime version to align operation versions

## Running Demo
Since a live demo uses OS's audio APIs, first, make sure you are NOT running on a docker container. Only audio APIs that support loopback recording are able to run locally (ie. Linux/Pulseaudio, Windows/WASAPI).
```shell
pip install SoundCard
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
python3 tensorflow/transducer_stream_demo.py
```