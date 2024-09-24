/**
* Get the model estimatived outputs for a ONE SAMPLE
* @param {Array} sample_inputs 
* @returns {Array}
*/
net.MLP.prototype.estimate_values = function( sample_inputs=[] ){

    let context = this; //The model context

    //Validations
    if( !(sample_inputs instanceof Array) ){
        throw Error(`The sample_inputs=${sample_inputs} need be a Array instance!`);
    }

    if( sample_inputs.length == 0 ){
        throw Error(`The sample_inputs is empty Array!`);
    }

    /**
    * Store the inputs of each layer (that all units of the layer receives)
    * And vinculate this object in the MLP(the father of the layers) to easy access and manipulations
    */
    let inputs_of_each_layer   = {};
    context.vinculate('inputs_of_each_layer', inputs_of_each_layer);

    /**
    * Store the estimatives of each unit of each layer 
    * And vinculate this object in the MLP(the father of the layers) to easy access and manipulations
    */
    let estimatives_of_each_layer  = {};
    context.vinculate('estimatives_of_each_layer', estimatives_of_each_layer);

    /**
    * The inputs of a layer <layer_index> is always the estimated values of previous layer( <layer_index> - 1 )
    * So, the property LAYER_INPUTS of the first hidden layer is the sample_inputs. And in the "estimate_values" phase, each layer(<layer_index>) will have the property LAYER_INPUTS, storing the estimated values of the previous layer(<layer_index> - 1) 
    * 
    * So, the inputs of first hidden layer( that is <layer_index>=0 ), will be the sample_inputs
    * And the inputs of secound hidden layer( that is <layer_index>=1 ), will be the estimated values of the first hidden layer( that is <layer_index>=0 )
    *
    * Always in this way.
    */
    context.get_first_hidden_layer()
           .setInputs( [... sample_inputs] );

    //The estimated values of OUTPUT LAYER
    let final_estimatives         = []; 

    /**
    * In this case, the layer 0 is the first hidden layer, because the input layer is ignored in initialization
    *
    * So
    * For each layer:
    */
    context.getLayers().forEach(function( current_layer, 
                                          layer_index 
    ){
        /**
        * For each unit in current layer, get the UNIT OUTPUT and store inside the unit
        */
        let units_estimatives = current_layer.get_Estimative_of_Units();

        /**
        * Store the outputs
        */
        estimatives_of_each_layer[ `layer${ layer_index }` ] = {};
        units_estimatives.forEach(function( unitOutput, index ){
            estimatives_of_each_layer[ `layer${ layer_index }` ][ `unit${ index }` ] = unitOutput;
        });
        
        /**
        * If the current layer is NOT the output layer
        */
        if( current_layer.notIs('output') ){

            /*
            * The inputs of a layer <layer_index> is always the estimated values of previous layer( <layer_index> - 1 ) 
            * Then the in lines below will Store the estimated values of the current layer( <layer_index> ) in the NEXT LAYER( <layer_index> + 1 ) AS UNIT_INPUTS
            */
            let next_layer = current_layer.getNextLayer();
        
            /**
            * Set the current layer( <layer_index> ) estimated values AS UNIT_INPUTS OF THE NEXT LAYER( <layer_index> + 1 )
            */
            next_layer.setInputs( units_estimatives );
        }

        /**
        * If is the output layer
        */
        if( current_layer.is('output') )
        {
            final_estimatives = units_estimatives;
        }

    });

    /**
    * Return the final estimatives( that is the estimated values of the output layer )
    */
    return {
        model_estimated_values : final_estimatives,
        inputs_of_each_layer    : inputs_of_each_layer,
        estimatives_of_each_layer   : estimatives_of_each_layer,

        /**
        * Get the estimated outputs
        * @returns {Array}
        */
        getEstimatedValues: function(){
            return final_estimatives;
        },

        /**
        * Get the input of each layer
        * @returns {Array}
        */
        getInputsOfEachLayer: function(){
            return inputs_of_each_layer;
        },

        /**
        * Get the output of each layer
        * @returns {Array}
        */
        getEstimativesOfEachLayer: function(){
            return estimatives_of_each_layer;
        }
    };
}