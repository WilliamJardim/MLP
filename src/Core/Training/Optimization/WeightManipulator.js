/**
* A utility for manipulate weights
* @param {Object} config 
*/
net.WeightManipulator = function( config ){
    let context = {};
    context._unit = config['context'];
    context.weightID = config['index'];
    context.value = context._unit.getWeightOfIndex(context.weightID);
    context.input = context._unit.getInputOfWeight(context.weightID);

    context.add = function( number ){
        context._unit.addWeight( context.weightID, number );
    }

    context.subtract = function( number ){
        context._unit.subtractWeight( context.weightID, number );
    }

    context.reset = function(){
        context._unit.setWeightOfIndex(context.weightID, 0);
    }

    return context;
}