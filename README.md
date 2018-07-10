# menoh
DNN interface library for NodeJS (powered by Menoh(c/c++) and MKL-DNN)

## Requirements
* MKL-DNN library [v0.14](https://github.com/intel/mkl-dnn/tree/v0.14) or later.
* ProtocolBuffers (Only tested with v3.5.1)
* [Menoh(C/C++) library](https://github.com/pfnet-research/menoh) v1.x (Only tested with v1.0.2)
* NodeJS v6 or greater

## Supported OS
* Mac
* Linux
* Windows

## Installation
Add `menoh` npm module to your project dependencies (package.json):
```
npm install menoh -S
```

### Installing dependencies
#### Mac & Linux
Simply follow the instruction described [here](https://github.com/pfnet-research/menoh/blob/v1.0.2/README.md).

For linux, you may need add `/usr/local/lib` to LD_LIBRARY_PATH depending on your linux distrubtion.
```sh
export LD_LIBRARY_PATH=/usr/local/lib
```

Or, you could add `/usr/local/lib` to [system library path](http://howtolamp.com/articles/adding-shared-libraries-to-system-library-path/).

#### Windows
You can download pre-build DLLs from [here](https://github.com/pfnet-research/menoh/releases/tag/v1.0.2).
The import library (menoh.lib) and its header files are bundled in this module and built during
its installation.
> Current version uses the import library built with *native* menoh v1.0.2.
    
You will need to copy the DLLs into a folder that is included in the `PATH` environment variable (e.g. C:Windows\\SysWOW64\\)

> The mklml.dll (included in the pre-built package for the native menoh v1.0.2) depends on `msvcr120.dll`. If
> your system does not have it, install [Visual C++ 2013 Redistibutable Package](https://support.microsoft.com/en-us/help/3179560/update-for-visual-c-2013-and-visual-c-redistributable-package).


## Run examples
Checkout the repository, cd into the root folder, then:
```
npm install
```

### VGG16 examples
```
cd example
sh retrieve_vgg16_data.sh
```

Then, run the VGG16 example.
```
node example_vgg16.js
```

You should see something similar to following:
```
### Result for ../test/data/Light_sussex_hen.jpg
fc6 out: -29.68303871154785 -52.6440544128418 0.9215406179428101 21.43817710876465 -6.305706977844238 ...
Top 5 categories are:
[8] 0.8902806639671326 n01514859 hen
[86] 0.037541598081588745 n01807496 partridge
[7] 0.03157550096511841 n01514668 cock
[82] 0.017570357769727707 n01797886 ruffed grouse, partridge, Bonasa umbellus
[83] 0.002043411135673523 n01798484 prairie chicken, prairie grouse, prairie fowl
### Result for ../test/data/honda_nsx.jpg
fc6 out: 14.704771041870117 -10.323609352111816 -32.17032241821289 -9.661919593811035 -14.448777198791504 ...
Top 5 categories are:
[751] 0.6547003388404846 n04037443 racer, race car, racing car
[817] 0.28364330530166626 n04285008 sports car, sport car
[573] 0.02763519063591957 n03444034 go-kart
[511] 0.01738707721233368 n03100240 convertible
[814] 0.004731603432446718 n04273569 speedboat
```

### MNIST examples
In the example folder...
```
$ node example_mnist.js
### Result for ../test/data/mnist/0.png
[0] 9792.962890625 Zero
### Result for ../test/data/mnist/1.png
[1] 4203.07470703125 One
### Result for ../test/data/mnist/2.png
[2] 7281.75341796875 Two
### Result for ../test/data/mnist/3.png
[3] 7360.65625 Three
### Result for ../test/data/mnist/4.png
[4] 3837.8447265625 Four
### Result for ../test/data/mnist/5.png
[5] 5259.931640625 Five
### Result for ../test/data/mnist/6.png
[6] 3743.64306640625 Six
### Result for ../test/data/mnist/7.png
[7] 4321.0859375 Seven
### Result for ../test/data/mnist/8.png
[8] 3331.339111328125 Eight
### Result for ../test/data/mnist/9.png
[9] 1424.4774169921875 Nine
```

> Read the comments in the examples for more details.

## API
```js
const menoh = require('menoh');
```

### Module methods
#### menoh.getNativeVersion() => {string}
Returns the version of underlying native menoh (core) library.

#### menoh.create(onnx_file_path{string}, [cb]) => {Promise}
Returns promise if `cb` is not provided. The promise resolves to a new instance of ModelBuilder.

### ModelBuilder methods
#### builder.addInput(input_var_name{string}, dims{array}) => {void}
Add an input profile for the given name.

#### builder.addOutput(output_var_name{string}) => {void}
Add an output profile for the given name.
> It currently takes no argument other than the name.

#### builder.buildModel(config{object}) => {Model}
Returns an executable model.
The config object can have two properties:
* backendName {string}: defaults to "mkldnn" or explicitly set it to "mkldnn" always.
* backendConfig {string}: a JSON string. defaults to "" or set to "" always.

You may build more than one model from the same builder.

### Model methods
#### model.setInputData(input_var_name{string}, data{array})
Sets input data for the give input name.

#### model.run(cb) => {Promise}
Run inference. It returns promise if `cb` is not provided. The actual inference takes place
in a background worker thread. You may run a different models concurrently to take advantage of
available CPU cores.

#### model.getOutput(output_var_name) => {object}
Returns output object generated during `model.run()` for the given output name.
The output object has following properties:
* dims {array}: Output data dimensions. (e.g. [1, 3, 244, 244])
* data {array}: Output data (flat array).


## Limitations
* You may not call `run()` on the *same model* more than once concurrently. The second run() will
fail with an error. Consider building another model for the concurrent operations.
