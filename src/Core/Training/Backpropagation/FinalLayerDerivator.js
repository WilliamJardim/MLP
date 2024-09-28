/**
* Calculate the derivative of each unit in the final layer
* 
* @param {Object} model_context                        - The model context
* @param {Array}  model_estimated_values               - The network estimations(of each unit)
* @param {Array}  desiredValuess                       - The desired values(for each unit)
* @param {Object} map_to_store_gradients               - A object to store the gradients of each unit with respect to each unit weight
*
* @returns {Object} - The calculated gradients of the final layer
* 
*/
net.MLP.prototype.FinalLayerDerivator = function( model_estimated_values, 
                                                  desiredValuess, 

                                                  //Lists to append the gradients
                                                  map_to_store_gradients
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
        //list_to_store_gradients_of_units[ `layer${ number_of_layers-1 }` ]    = {};
        //list_to_store_gradients_for_weights[ `layer${ number_of_layers-1 }` ] = {};

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
            let derivative   = estimationDifference * estimatedValueDerivative;

            /**
            * Compute and store the gradients of this unit
            */
            map_to_store_gradients.storeGradientOf({
                                    //Unit data
                                    derivative,
                                    ofUnit  : final_unit_index, 

                                    //Layer data
                                    ofLayer : index_of_last_layer
                                });

        });

        return map_to_store_gradients;
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