/**
* A utility for manipulate weights
* @param {Object} config 
*/
net.WeightManipulator = function( config ){
    let context = {};
    context._unit = config['context'];
    context.weightID = config['index'];
    context.value = context._unit.getWeight(context.weightID);
    context.input = context._unit.getInputOfWeight(context.weightID);

    context.add = function( number ){
        context._unit.weights[ context.weightID ] = context._unit.weights[ context.weightID ] + number;
    }

    context.subtract = function( number ){
        context._unit.weights[ context.weightID ] = context._unit.weights[ context.weightID ] - number;
    }

    context.reset = function(){
        context._unit.weights[ context.weightID ] = 0;
    }

    return context;
}