/**
* Do the backpropagation step for ONE SAMPLE 
* 
* This backpropagation implementation was inspired by various publicly available resources, including concepts 
* and explanations from the work of Jason Brownlee on backpropagation.
* 
* CREDITS && REFERENCE:
* Jason Brownlee, How to Code a Neural Network with Backpropagation In Python (from scratch), Machine Learning Mastery, Available from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/, accessed April 15th, 2024.
* 
*
* @param {Array} sample_inputs  - the sample features
* @param {Array} desiredOutputs - the DESIRED values of the last layer units
* 
* @returns {Object} - The mapped gradients of the units of each layer AND The mapped gradients of the each weight of each unit of each layer
*/
net.MLP.prototype.backpropagate_sample = function( sample_inputs  = [], 
                                                   desiredOutputs = [] 
){

    let context = this; //The model context

    //Validations
    if( !(sample_inputs instanceof Array) ){
        throw Error(`The sample_inputs=[${sample_inputs}] need be a Array instance!`);
    }

    if( sample_inputs.length == 0 ){
        throw Error(`The sample_inputs is empty Array!`);
    }

    if( !(desiredOutputs instanceof Array) ){
        throw Error(`The desiredOutputs=[${desiredOutputs}] need be a Array instance!`);
    }

    if( desiredOutputs.length == 0 ){
        throw Error(`The desiredOutputs is empty Array!`);
    }

    let number_of_layers         = context.getLayers().length;

    //Get the model estimated values
    let estimated_data         = context.estimate_values( sample_inputs );

    //Extract the data from the model estimated values
    let model_estimated_values  = estimated_data.getEstimatedValues();
    let inputs_of_each_layer     = estimated_data.getInputsOfEachLayer();
    let estimatives_of_each_layer    = estimated_data.getEstimativesOfEachLayer();

    /**
    * List store the gradients of each unit of the all layers
    * Format { layer_number: gradients_object ...}
    */
    let list_to_store_gradients_of_units = {};

    /**
    * List store the gradients for the all layers(of each weight of each unit)
    * Format { layer_number: [ unit: { weight: [] ...} ] ...}
    */
    let list_to_store_gradients_for_weights = {};

    /**
    * Calculate the derivative of each unit in last layer
    * This process is made by a subtraction of the "unit estimated value" and the "desired value for the unit".
    * So, Each unit have a "desired value", and each unit produces a "estimative" in the "estimate_values" phase, so these informations are used to calculate this derivatives
    * 
    * And these gradients will be stored in the list_to_store_gradients_of_units and list_to_store_gradients_for_weights
    */
    context.calculate_derivatives_of_last_layer_units( model_estimated_values, 
                                                   desiredOutputs, 
                                                   list_to_store_gradients_of_units, 
                                                   list_to_store_gradients_for_weights );

    /** 
    * Start the backpropagation
    * Starting in LAST HIDDEN LAYER and going in direction of the FIRST HIDDEN LAYER.
    * 
    * When we reach the FIRST HIDDEN LAYER, when had calculate the gradients of all units in the FIRST HIDDEN LAYER, the backpropagation finally ends.
    */
    let currentLayerIndex = number_of_layers-1-1;

    /**
    * While the "while loop" not arrived the first hidden layer
    * The first hidden layer(that have index 0, will be the last layer that will be computed) 
    */
    while( currentLayerIndex >= 0 )
    {
        //Current layer data
        let current_layer          = context.getLayer( currentLayerIndex );
        let current_layer_inputs   = inputs_of_each_layer[ `layer${ currentLayerIndex }` ];
        let current_layer_estimatives  = estimatives_of_each_layer[ `layer${ currentLayerIndex }` ]

        list_to_store_gradients_of_units[ `layer${ currentLayerIndex }` ]     = {};
        list_to_store_gradients_for_weights[ `layer${ currentLayerIndex }` ]  = {};

        /**
        * Next layer data:
        */
        let next_layer_index               = currentLayerIndex + 1;
        let next_layer                     = context.getLayer( next_layer_index );
        let next_layer_units               = next_layer.getUnits();

        /**
        * Get the gradients(of the units) of the next layer
        * These gradients will be used in lines below:
        */
        let next_layer_gradients           = list_to_store_gradients_of_units[ `layer${ next_layer_index }` ];

        /**
        * For each unit in CURRENT HIDDEN LAYER
        */
        current_layer.getUnits().forEach(function( current_hidden_layer_unit, 
                                                   the_unit_index
        ){

            let hidden_unit_index           = the_unit_index; //I also will call as UH, that is The index of the current unit, like in the equation above;
            let current_unit_weights        = current_hidden_layer_unit.getWeights();
            let current_unit_estimative         = current_layer_estimatives[ `unit${ the_unit_index }` ];
            let current_unit_function_name  = current_hidden_layer_unit.getFunctionName();

            /**
            * I make a copy of the "current_layer_inputs" variable to make a safe and independent copy of the same layer inputs for all units,
            * 
            * Because, the key point of this is: 
            * 
            *    All units receives the same inputs!. That is, the same inputs of the layer Who owns the unit
            *    Because, each layer have N inputs. And the inputs of EACH UNIT in a layer are the estimated values of the previous layer.
            *    So, in short, in a given layer of the neural network, all units in that layer will receive exactly the same inputs, THAT IS, THE OUTPUT OF THE PREVIOUS LAYER, WHICH ARE ITS INPUTS   
            *
            *    Except the input layer, because the input layer has no units, It only has inputs(just numbers), and nothing more than that.
            *    Therefore, the input layer has no units, and therefore does not receive input from a previous layer
            *
            *    To facilitate and standardize the process, in a generic way for each layer, we can imagine that the estimated values of the previous layer are the inputs themselves.               
            */
            let current_unit_inputs       = [... current_layer_inputs.copyWithin()];

            /**
            * Calculate the derivative of the current unit UH
            * Relembering that, 
            * The derivative of a unit in a hidden layer will always depend of the derivatives of the next layer
            */ 
            context.calculate_hidden_unit_derivative( 
                    index_of_current_hidden_layer        = currentLayerIndex,
                    current_hidden_unit_index            = hidden_unit_index, 
                    weights_of_current_hidden_unit       = current_unit_weights,
                    current_unit_inputs_values           = current_unit_inputs,
                    current_unit_function_name           = current_unit_function_name,
                    current_unit_estimative_value            = current_unit_estimative,
                    next_layer_units_objects             = next_layer_units,
                    next_layer_units_gradients           = next_layer_gradients,

                    list_to_store_gradients_of_units     = list_to_store_gradients_of_units,
                    list_to_store_gradients_for_weights  = list_to_store_gradients_for_weights
            );
        });

        /**
        * Goto previous layer
        */
        currentLayerIndex--;
    }

    /**
    * Return the calculated gradients for the sample
    */
    return {
        gradients_of_units          : list_to_store_gradients_of_units,
        gradients_for_each_weights  : list_to_store_gradients_for_weights
    };
}