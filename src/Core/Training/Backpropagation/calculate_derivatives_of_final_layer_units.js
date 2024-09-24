/**
* Calculate the derivative of each unit in the final layer
* 
* @param {Array}  model_estimated_values              - The network estimations(of each unit)
* @param {Array}  desiredValuess                       - The desired values(for each unit)
* @param {Object} list_to_store_gradients_of_units     - A object to store the gradients of each unit
* @param {Object} list_to_store_gradients_for_weights  - A object to store the gradients of each unit with respect to each unit weight
*
* @returns {Object} - The calculated gradients of the final layer
* 
*/
net.MLP.prototype.calculate_derivatives_of_final_layer_units = function(model_estimated_values, 
                                                                       desiredValuess, 
                                                                       list_to_store_gradients_of_units, 
                                                                       list_to_store_gradients_for_weights 
){
    let context = this; //The model context
    
    /**
    * Create a derivator for derivate the final layer 
    */
    let derivatorInstance = net.FinalLayerDerivator(context, 
                                                    model_estimated_values, 
                                                    desiredValuess, 
                                                    list_to_store_gradients_of_units, 
                                                    list_to_store_gradients_for_weights);

    /**
    * Get and return the gradients of the last model layer 
    */
    let lastLayerGradients = derivatorInstance.derivate();
    return lastLayerGradients;
}