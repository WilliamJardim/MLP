/**
* Do the backpropagation step for ONE SAMPLE 
* This implementation is entirely original, written from scratch in JavaScript.
*
* This implementation was inspired by various publicly available resources, including concepts 
* and explanations from the work of Jason Brownlee on backpropagation.
* 
* CREDITS && REFERENCE:
* Jason Brownlee, How to Code a Neural Network with Backpropagation In Python (from scratch), Machine Learning Mastery, Available from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/, accessed April 15th, 2024.
* 
*
* @param {Array} sample_inputs  - the sample features
* @param {Array} desiredValuess - the DESIRED values of the final layer units
* 
* @returns {Object} - The mapped gradients of the units of each layer AND The mapped gradients of the each weight of each unit of each layer
*/
net.MLP.prototype.backpropagate_sample = function({
                                                    sample_inputs  = [], 
                                                    desiredValuess = [],
                                                    beforeThis = ()=>{},
                                                    afterThis=()=>{}
}){
    //Validations
    if( !(sample_inputs instanceof Array) ){
        throw Error(`The sample_inputs=[${sample_inputs}] need be a Array instance!`);
    }

    if( sample_inputs.length == 0 ){
        throw Error(`The sample_inputs is empty Array!`);
    }

    if( !(desiredValuess instanceof Array) ){
        throw Error(`The desiredValuess=[${desiredValuess}] need be a Array instance!`);
    }

    if( desiredValuess.length == 0 ){
        throw Error(`The desiredValuess is empty Array!`);
    }

    let modelContext                 = this; //The model context
    let model_estimatives_data       = modelContext.estimate_values( sample_inputs ); 
    let model_estimatives            = model_estimatives_data.getEstimatedValues();        //Get propagation values
    let estimatives_of_each_layer    = model_estimatives_data.getEstimativesOfEachLayer(); //Get the estimative-values of each unit of each layer
    let inputs_of_each_layer         = model_estimatives_data.getInputsOfEachLayer();

    /**
    * Store the gradients of the weights and bias, each unit of each layer 
    * 
    * "gradients_per_layer" structure: 
    * layer
    *    --> unit == GradientVector
    * 
    * That is, the "gradients_per_layer" contains "layers" as keys, and the "layers" contains "units" as keys, and the "units" is GradientVector instances
    */
    let gradients_per_layer = modelContext.readProp('gradients_per_layer');

    /** 
    * Clean all, the clear old gradients  
    */
    gradients_per_layer.eraseAll();

    if( beforeThis )
    {
        beforeThis.bind(modelContext)({
            modelContext,
            model_estimatives_data,
            model_estimatives,
            estimatives_of_each_layer,
            inputs_of_each_layer
        });
    }

    /** 
    * Calculate the gradients of the final layer, and store in the "gradients_per_layer"
    */
    modelContext.calculate_gradients_of_final_layer({
        modelContext, 
        inputs_of_each_layer, 
        model_estimatives, 
        desiredValuess, 
        gradients_per_layer
    });

    /** 
    * Start the backpropagation
    * Starting in LAST HIDDEN LAYER and going in direction of the FIRST HIDDEN LAYER.
    * 
    * When we reach the FIRST HIDDEN LAYER, when had calculate the gradients of all units in the FIRST HIDDEN LAYER, the backpropagation finally ends.
    */

    let number_of_layers = modelContext.number_of_layers - 1 - 1 - 1; //Ignore the input layer and ignore the final layer

    /**
    * While the "while loop" not arrived the first hidden layer
    * The first hidden layer(that have index 0, will be the final layer that will be computed) 
    */
    let current_hidden_layer = number_of_layers;
    while( current_hidden_layer >= 0 )
    {
        /** Calculate the gradients of the current hidden layer  */
        modelContext.calculate_gradients_of_a_hidden_layer({
            modelContext, 
            inputs_of_each_layer, 
            estimatives_of_each_layer, 
            current_hidden_layer, 
            gradients_per_layer
        });

        /** Do the same for the Previous layer **/
        current_hidden_layer--;

    }

    /**
    * Split the gradients in two distinct objects 
    */
    let gradients_of_each_unit_bias_per_layer = {};
    let gradients_of_each_unit_weights_per_layer = {};
    
    let layersKeys = Object.keys( gradients_per_layer.table );
    layersKeys.forEach((layerId)=>{
        
        let unitsKeys = Object.keys( gradients_per_layer[layerId].table );
        gradients_of_each_unit_bias_per_layer[ layerId ] = {};
        gradients_of_each_unit_weights_per_layer[ layerId ] = {};

        unitsKeys.forEach((unitId)=>{
            gradients_of_each_unit_bias_per_layer[ layerId ][ unitId ]  = gradients_per_layer[layerId][unitId].loss_wrt_unit_estimation;
            gradients_of_each_unit_weights_per_layer[ layerId ][ unitId ] = gradients_per_layer[layerId][unitId].loss_wrt_unit_weights;
        });
    });

    if( afterThis )
    {
        afterThis.bind(modelContext)({
            modelContext,
            gradients_per_layer,
            gradients_of_each_unit_weights_per_layer,
            gradients_of_each_unit_bias_per_layer,
            model_estimatives_data,
            model_estimatives,
            estimatives_of_each_layer,
            inputs_of_each_layer
        });
    }

    /**
    * Return the gradients
    */
    return {
        gradients_per_layer,
        gradients_of_each_unit_weights_per_layer,
        gradients_of_each_unit_bias_per_layer
    };
    
}