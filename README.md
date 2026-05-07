# diff-chunk-demo

A website to demo various language model's speed + accuracy.

There are four models: two with the standard transformer architecture and two with a custom architecture.
One of each type has context length 256, and the other has context length 1024.

The website uses PyScript to run the models with the JavaScript onnxruntime API.
