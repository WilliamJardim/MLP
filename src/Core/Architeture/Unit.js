/**
* Multilayer Perceptron Neural Network (MLP)
* By William Alves Jardim
* 
* This implementation is entirely original, written from scratch in JavaScript.
* It was inspired by various publicly available resources, including concepts 
* and explanations from the work of Jason Brownlee on backpropagation.
* 
* CREDITS && REFERENCE:
* Jason Brownlee, How to Code a Neural Network with Backpropagation In Python (from scratch), Machine Learning Mastery, Available from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/, accessed April 15th, 2024.
* 
* For more details, see README.md.
*/

//A Unit(with just estimateOutput and weight initialization)
net.Unit = function( unit_config={} ){
    let context = {};

    context.objectName            = 'Unit';

    //Parameters
    context.number_of_inputs      = unit_config.number_of_inputs     || Number();
    context.activation_function   = unit_config.activation_function  || 'sigmoid';
    context.weights               = unit_config.weights              || Array(context.number_of_inputs).fill(0);
    context.bias                  = unit_config.bias                 || Number();

    /**
    * Get the layer object( That was linked to this unit )
    * @returns {Object}
    */
    context.getLayerOfThisUnit = function(){
        return context._layerRef; 
    }

    /**
    * Vinculate a prop to this unit
    * @param {String} newAttributeName
    * @param {any}    valueOfThisAttribute
    */
    context.vinculate = function(newAttributeName, valueOfThisAttribute){
        context[ newAttributeName ] = valueOfThisAttribute;
    }

    /**
    * Get the activation function name
    */
    context.getFunctionName = function(){
        return context.activation_function;
    }

    //Setters
    context.setWeights = function( newWeights=[] ){
        if( !(newWeights instanceof Array) ){
            throw Error(`the newWeights=${newWeights} is not a Array instance!`);
        }
        if( newWeights.length == 0 ){
            throw Error(`the newWeights is empty Array!`);
        }
        context.weights = newWeights;
    }

    //Getter for the context.weights
    context.getWeights = function(){
        return context.weights;
    }

    //Add a value to a especific weight
    context.addWeight = function( weight_index, number ){
        context.weights[ weight_index ] = context.weights[ weight_index ] + number;
    }

    //Subtract a value to a especific weight
    context.subtractWeight = function( weight_index, number ){
        context.weights[ weight_index ] = context.weights[ weight_index ] - number;
    }

    //Get a WeightManipulator for a especific weight
    context.selectWeight = function( weight_index ){
        return net.WeightManipulator({
            context: context,
            index: weight_index
        });
    }

    //Setter for bias
    context.setBias = function( newBias=Number() ){
        if( typeof newBias != 'number' ){
            throw Error(`the newBias=${newBias} is not a Number instance!`);
        }
        context.bias = newBias;
    }

    //Add a value for the bias
    context.addBias = function( number ){
        context.bias = context.bias + number;
    }

    //Subtract a value for the bias
    context.subtractBias = function( number ){
        context.bias = context.bias - number;
    }

    /**
    * Generate random parameters
    */
    context.generate_random_parameters = function( number_of_inputs=context.number_of_inputs ){

        for( let i = 0 ; i < number_of_inputs ; i++ )
        {
            context.weights[i] = Math.random();
        }

        context.bias = Math.random();
    }

    /**
    * Get the estimated output for ONE SAMPLE of this UNIT, to get the estimated output
    * @param   {Array}   sample_inputs        - The model inputs
    * @returns {Number}                       - The model estimative
    */
    context.estimateOutput = function( sample_inputs )
    {
        let number_of_inputs = sample_inputs.length;

        let summed_value = 0;

        for( let i = 0 ; i < number_of_inputs ; i++ ) 
        {
            let weight_index = i;
            summed_value = summed_value + ( sample_inputs[i] * context.getWeight( weight_index ) );
        }

        //Add the bias
        summed_value = summed_value + context.bias;

        let output = net.activations[ context.getFunctionName() ]( summed_value );

        return {
            activation_function_output: output,
            unit_potential: summed_value //The unit activation potential(just the summed value)
        };
    }

    /**
    * Get a especific weight of this unit 
    * @param {Number} weight_index
    * @returns {Number}
    */
    context.getWeight = function( weight_index ){
        return context.weights[ weight_index ]
    }

    /**
    * Get a especific input of weight
    * @param {Number} weight_index 
    * @returns {Number}
    */
    context.getInputOfWeight = function( weight_index ){
        let father_layer          = context.getLayerOfThisUnit(), //Get the layer to which THIS UNIT belongs
            father_layer_inputs   = father_layer.getInputs();     //The inputs of the layer(that is, the layer to which THIS UNIT belongs)

        return father_layer_inputs[ weight_index ];
    }

    return context;
}