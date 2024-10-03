net.MLP.prototype.calculate_gradients_of_final_layer = function( { modelContext, inputs_of_each_layer, model_estimatives, desiredValuess, gradients_per_layer } )
{
    /**
    * Calculate the derivative of each unit in final layer
    * This process is made by a subtraction of the "unit estimated value" and the "desired value for the unit".
    * So, Each unit have a "desired value", and each unit produces a "estimative" in the "estimate_values" phase, so these informations are used to calculate this derivatives
    * 
    * And these gradients will be stored in the "gradients_per_layer"
    *
    * So to do this, i will use the following below:
    */
    let final_layer_index            = modelContext.last_layer_index - 1;

    gradients_per_layer.registryLayer( final_layer_index );

    let number_of_final_layer_units  = modelContext.last_layer.units;
    
    let unit_number = 0;
    while( unit_number < number_of_final_layer_units )
    {
        let unit_obj           = modelContext.getLayer(final_layer_index)
                                             .getUnit( unit_number );

        let unit_inputs        = inputs_of_each_layer[ `layer${ final_layer_index }` ];

        let unit_function      = net.activations[ unit_obj.activation_function ];
        let estimative         = model_estimatives[ unit_number ];
        let desired_value      = desiredValuess[ unit_number ];
        
        let loss_wrt_unit_estimation = ( estimative - desired_value ) * unit_function.derivative( estimative );

        /**
        * Compute the partial derivatives of each weight parameter( that is the gradient vector ), and store as GradientVector instance
        */
        gradients_per_layer.setGradientWrtOf({
            ofUnit       : unit_number,
            ofLayer      : final_layer_index,

            setGradient  : new net.GradientVector({
                //Repass the LOSS WRT OF THE UNIT ESTIMATION FUNCTION
                loss_wrt_unit_estimation,

                //Get the inputs of the weights of the current unit
                unit_inputs
            })
        });
                                
        /** DO THE SAMES FOR THE NEXT UNIT IN THE CURRENT HIDDEN LAYER  */
        unit_number++;
    }

    /**
    * Return the "gradients_per_layer", with the gradients of the final layer already calculated
    */
    return gradients_per_layer;

}