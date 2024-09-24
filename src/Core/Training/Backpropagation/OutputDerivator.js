/**
* Calculate the derivative of each unit in the output layer
* 
* @param {Object} model_context                        - The model context
* @param {Array}  model_estimated_values              - The network estimations(of each unit)
* @param {Array}  desiredOutputs                       - The desired values(for each unit)
* @param {Object} list_to_store_gradients_of_units     - A object to store the gradients of each unit
* @param {Object} list_to_store_gradients_for_weights  - A object to store the gradients of each unit with respect to each unit weight
*
* @returns {Object} - The calculated gradients of the output layer
* 
*/
net.OutputDerivator = function( model_context,
                                model_estimated_values, 
                                desiredOutputs, 
                                //Lists to append the gradients
                                list_to_store_gradients_of_units, 
                                list_to_store_gradients_for_weights
){
    let context = {};
    context.model_context = model_context;
    context.model_estimated_values = model_estimated_values;
    context.desiredOutputs = desiredOutputs;
    context.list_to_store_gradients_of_units = list_to_store_gradients_of_units;
    context.list_to_store_gradients_for_weights = list_to_store_gradients_for_weights;

    context.derivate = function(){

        let number_of_layers         = context.model_context.getLayers().length;
        let index_of_last_layer    = number_of_layers-1;

        //Registry a object for store the gradients of the units of the output layer
        context.list_to_store_gradients_of_units[ `layer${ number_of_layers-1 }` ] = {};
        context.list_to_store_gradients_for_weights[ `layer${ number_of_layers-1 }` ] = {};

        /**
        * Calculate the LOSS derivative of each output unit, using the chain rule of calculus
        * That is, the derivative of the LOSS with respect to estimated value of the output unit
        * 
        * For calculate the LOSS derivative of the each output unit, The formula used is:
        *  
        *   unit<N>_derivative = ( output_unit<N>_estimated_value - output_unit<N>_desired_value ) * derivative_of_function_of_output_unit<N>
        *
        * Like he can see below:
        */
        context.model_context
        .getOutputLayer()
        .getUnits()
        .forEach(function( output_unit, 
                           output_unit_index 
        ){

            let unitActivationFn     = output_unit.getFunctionName();

            let unitOutput           = context.model_estimated_values[ output_unit_index ];

            let desiredOutput        = context.desiredOutputs[ output_unit_index ];

            let outputDifference     = unitOutput - desiredOutput;

            //The activation function of this U output unit
            let unit_function_object = net.activations[ unitActivationFn ];

            //The derivative of activation funcion of this U output unit(at output layer)
            let outputDerivative  = unit_function_object.derivative( unitOutput );

            //The derivative of this output unit U
            let unit_derivative   = outputDifference * outputDerivative;

            //Store the gradient in the gradients object
            context.list_to_store_gradients_of_units[ `layer${ index_of_last_layer }` ][ `unit${ output_unit_index }` ] = unit_derivative;

            //Store the gradient with respect of each weight
            context.list_to_store_gradients_for_weights[ `layer${ index_of_last_layer }` ][ `unit${ output_unit_index }` ] = [];

            //For each weight
            output_unit.getWeights().forEach(function(weight_value, weight_index_c){

                let weight_input_C = output_unit.getInputOfWeight( weight_index_c );  

                context.list_to_store_gradients_for_weights[ `layer${ index_of_last_layer }` ][ `unit${ output_unit_index }` ][ weight_index_c ] = unit_derivative * weight_input_C;

            });

        });

        return {
            calculated_gradients_of_units: {... JSON.parse(JSON.stringify(context.list_to_store_gradients_of_units)) },
            calculated_gradients_for_weights: {... JSON.parse(JSON.stringify(context.list_to_store_gradients_for_weights)) }
        };
    }

    return context;
}