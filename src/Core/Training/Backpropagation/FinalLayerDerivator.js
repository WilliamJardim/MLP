/**
* Calculate the derivative of each unit in the final layer
* 
* @param {Object} model_context                        - The model context
* @param {Array}  model_estimated_values               - The network estimations(of each unit)
* @param {Array}  desiredValuess                       - The desired values(for each unit)
* @param {Object} list_to_store_gradients_of_units     - A object to store the gradients of each unit
* @param {Object} list_to_store_gradients_for_weights  - A object to store the gradients of each unit with respect to each unit weight
*
* @returns {Object} - The calculated gradients of the final layer
* 
*/
net.MLP.prototype.FinalLayerDerivator = function( model_estimated_values, 
                                                  desiredValuess, 
                                                  //Lists to append the gradients
                                                  list_to_store_gradients_of_units, 
                                                  list_to_store_gradients_for_weights
){
    let context = this; //The model context
    let model_context = context; //Alias for the model context
    
    let derivation_context = {}; //The sub private context to be used to store the values that will used in derivation

    /**
    * Declare the derivative function
    */
    derivation_context.do_derivative = function(){

        let number_of_layers       = model_context.getLayers().length;
        let index_of_last_layer    = number_of_layers-1;

        //Registry a object for store the gradients of the units of the final layer
        list_to_store_gradients_of_units[ `layer${ number_of_layers-1 }` ]    = {};
        list_to_store_gradients_for_weights[ `layer${ number_of_layers-1 }` ] = {};

        /**
        * Calculate the LOSS derivative of each final layer unit, using the chain rule of calculus
        * That is, the derivative of the LOSS with respect to estimated value of the final layer unit
        * 
        * For calculate the LOSS derivative of the each final layer unit, The formula used is:
        *  
        *   unit<N>_derivative = ( final_unit<N>_estimated_value - final_unit<N>_desired_value ) * derivative_of_function_of_final_unit<N>
        *
        * Like he can see below:
        */
        model_context
        .atSelf()
        .getFinalLayer()
        .getUnits()
        .forEach(function( final_unit, 
                           final_unit_index 
        ){

            let unitActivationFn      = final_unit.getFunctionName();

            let unitEstimatedValue    = model_estimated_values[ final_unit_index ];

            let desiredValues         = desiredValuess[ final_unit_index ];

            let estimationDifference  = unitEstimatedValue - desiredValues;

            //The activation function of this U final unit
            let unit_function_object   = net.activations[ unitActivationFn ];

            //The derivative of activation funcion of this U final unit(at final layer)
            let estimatedValueDerivative  = unit_function_object.derivative( unitEstimatedValue );

            //The derivative of this final unit U
            let unit_derivative   = estimationDifference * estimatedValueDerivative;

            //Store the gradient in the gradients object
            list_to_store_gradients_of_units[ `layer${ index_of_last_layer }` ][ `unit${ final_unit_index }` ] = unit_derivative;

            //Store the gradient with respect of each weight
            list_to_store_gradients_for_weights[ `layer${ index_of_last_layer }` ][ `unit${ final_unit_index }` ] = [];

            //For each weight
            final_unit.getWeights().forEach(function(weight_value, weight_index_c){

                let weight_input_C = final_unit.getInputOfWeight( weight_index_c );  

                list_to_store_gradients_for_weights[ `layer${ index_of_last_layer }` ][ `unit${ final_unit_index }` ][ weight_index_c ] = unit_derivative * weight_input_C;

            });

        });

        return {
            calculated_gradients_of_units: {... JSON.parse(JSON.stringify(list_to_store_gradients_of_units)) },
            calculated_gradients_for_weights: {... JSON.parse(JSON.stringify(list_to_store_gradients_for_weights)) }
        };
    }

    /**
    * The FinalLayerDerivator it self
    * @returns {Object}
    */
    derivation_context.atSelf = function(){
        return derivation_context;
    }

    /**
    * Returns the derivator object, that will be used in backpropagation 
    */
    return derivation_context;
}