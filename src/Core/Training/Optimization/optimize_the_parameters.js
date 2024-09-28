/**
* Applies the Gradient Descent algorithm 
* Is used to update the weights and bias of each unit in each layer
* 
* @param {Object} gradient_storage - The gradients of each weight and bias of each unit of each layer
*/
net.MLP.prototype.optimize_the_parameters = function( gradient_storage ){
    let context = this; //The model context

    //For each layer
    context.getLayers().forEach(function( current_layer, 
                                          layer_index                                       
    ){

        //For each unit in current layer
        current_layer.getUnits().forEach(function( current_unit, 
                                                   unit_index
        ){

            //For each weight
            current_unit.getWeights().forEach(function( weight_value, 
                                                        weight_index
            ){

                let calculated_gradients_values_for_weight = gradient_storage.getTableOfLayer( layer_index )
                                                                             .getGradientOfAUnit( unit_index )
                                                                             .getDerivativeOfWeight( weight_index );

                //Select this weight <weight_index> and update then
                current_unit.selectWeight( weight_index )
                            .subtract( context.learning_rate * calculated_gradients_values_for_weight );

            });

            let calculated_gradients_values_for_bias = gradient_storage.getTableOfLayer( layer_index )
                                                                       .getGradientOfAUnit( unit_index )
                                                                       .getDerivativeOfBias();

            //Update bias
            current_unit.subtractBias( context.learning_rate * calculated_gradients_values_for_bias );

        });

    });
}