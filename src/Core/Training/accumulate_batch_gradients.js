/**
* Acculumate the gradients of a full-batch(that is, of each sample in training_samples), and update the parameters
* This method will be used in the "fullbatch_train" training metheod
* 
* @param {Array} train_samples - The training samples
*/
net.MLP.prototype.accumulate_batch_gradients = function( train_samples ){
    let modelContext = this; //The model context
    
    let number_of_samples = train_samples.length;

    /**
    * The variable summed_gradients_for_weights, is used for:
    * Accumulate the gradient of each weight of each unit of each layer
    * 
    * Structure in visually representation is:
    *       layer0
    *          --> unit0
    *              --> accumulation_for_weight_0
    *              --> accumulation_for_weight_1
    *              --> accumulation_for_weight_N
    *                  (etc.. other accumulations)
    *                  (all the "accumulation_for_weight_N" is a Number) 
    * 
    *              (etc... other units)
    * 
    *      (etc... other layers)
    * 
    * Text description of this representation:
    * The variable summed_gradients_for_weights is a hashmap with layers.
    * Each layer have N units, and each unit have N "weight accumulation" or also called "accumulation_for_weight" in this text example, and are Numbers.
    * 
    * This is a accumulation of the gradient of each weight of each unit of each layer
    * The accumulation will be done in the lines bellow, using some for loops:
    */
    let summed_gradients_for_weights = {};

    /**
    * The variable summed_gradients_for_weights, is used for:
    * Accumulate the gradient of the bias of each unit of each layer
    * 
    * Structure in visually representation is:
    *       layer0
    *          --> accumulation_for_bias_of_unit_0  
    *          --> accumulation_for_bias_of_unit_1         
    *           (etc.. other weights gradients)
    *           (all the "bias_of_unit<N>" is a Number) 
    *
    *      (etc... other layers)
    * 
    * Text description of this representation:
    * The variable summed_gradients_for_bias is a hashmap with layers.
    * Each layer have N "bias accumulation"(corresponding to each unit) or also called "accumulation_for_bias_of_unit_<N>" in this text example, and is a Number.
    * 
    * This is a accumulation of the gradient of each the bias of each unit of each layer
    * The accumulation will be done in the lines bellow, using some for loops:
    */
    let summed_gradients_for_bias = {};

    /**
    * For each sample 
    * Will accumulating the gradients(of all samples)
    */
    train_samples.forEach(function( sample_data ){

        let sample_features         = sample_data[0]; //SAMPLE FEATURES
        let sample_desired_value    = sample_data[1]; //SAMPLE DESIRED VALUES

        //Validations before apply the backpropagation
        if( !(sample_features instanceof Array) ){
            throw Error(`The variable sample_features=[${sample_features}] must be a Array!`);
        }

        if( !(sample_desired_value instanceof Array) ){
            throw Error(`The variable sample_desired_value=[${sample_desired_value}] is not a Array!`);
        }

        //If the number of items in the sample_desired_value Array is different from the number of units in the final layer
        if( sample_desired_value.length != modelContext.layers[ modelContext.layers.length-1 ].units.length ){
            throw Error(`The sample_desired_value=[${sample_desired_value}] has ${sample_desired_value.length} elements, But must be ${modelContext.layers[ modelContext.layers.length-1 ].units.length}(the number of units in final layer)`);
        }

        //Do backpropagation and retrive the gradients
        let sample_gradients_data = modelContext.backpropagate_sample({
            sample_inputs  : sample_features, 
            desiredValuess : sample_desired_value,

            /**
            * Callback called after the "backpropagate_sample" execution
            * Use this callback to accumulate the gradients
            * 
            * Is full sequential and sync metheod, is not assync.
            */
            afterThis({ modelContext,
                        gradients_per_layer,
                        gradients_of_each_unit_weights_per_layer,
                        gradients_of_each_unit_bias_per_layer
            }){
                let sample_gradients_for_weights = gradients_of_each_unit_weights_per_layer;
                let sample_gradients_for_bias    = gradients_of_each_unit_bias_per_layer;

                //Accumulate the gradients
                let layersIds = Object.keys(sample_gradients_for_weights);

                layersIds.forEach(function(layerId){
                    let layerData  = sample_gradients_for_weights[ layerId ];
                    let unitsId    = Object.keys(layerData);

                    //If not existis the layer in summed_gradients_for_weights, create with empty object
                    if( summed_gradients_for_weights[ layerId ] == undefined ){
                        summed_gradients_for_weights[ layerId ] = {};
                    }

                    if( summed_gradients_for_bias[ layerId ] == undefined ){
                        summed_gradients_for_bias[ layerId ] = {};
                    }

                    unitsId.forEach(function(unitId){
                        let number_of_weights = sample_gradients_for_weights[ layerId ][ unitId ].length;

                        //If not exists the unit in summed_gradients_for_weights, create with zeros
                        if( summed_gradients_for_weights[ layerId ][ unitId ] == undefined ){
                            summed_gradients_for_weights[ layerId ][ unitId ] = Array(number_of_weights).fill(0);
                        }

                        //If not exists the unit in summed_gradients_for_bias, create with zero
                        if( summed_gradients_for_bias[ layerId ][ unitId ] == undefined ){
                            summed_gradients_for_bias[ layerId ][ unitId ] = 0;
                        }

                        /**
                        * I do the accumulation in the following way: 
                        * I sum all the gradient of all the weights( of each unit of each layer ), 
                        * in its corresponding position in the hashmap, that is, sample_gradients_for_weights[ layer<LAYER_INDEX> ] [ unit<UNIT_INDEX> ] [ <WEIGHT_INDEX> ] 
                        *
                        * Because this, The format of the result of this sum will be the same format of the "sample_gradients_for_weights" returned by the backpropagate_sample metheod
                        */
                        for( let c = 0 ; c < number_of_weights ; c++ )
                        {   
                            let weight_index = c;
                            let gradient_of_weight = sample_gradients_for_weights[ layerId ][ unitId ][ weight_index ];
                            summed_gradients_for_weights[ layerId ][ unitId ][ weight_index ] = summed_gradients_for_weights[ layerId ][ unitId ][ weight_index ] + gradient_of_weight;
                        }

                        //Sum gradient for acculate the bias(that no have inputs)
                        let gradient_of_bias = sample_gradients_for_bias[ layerId ][ unitId ];
                        summed_gradients_for_bias[ layerId ][ unitId ] = summed_gradients_for_bias[ layerId ][ unitId ] + gradient_of_bias;

                    });
                });
            }
        });

    });

    /** BELOW: Do the mean of the gradients of each weight(of each unit of each layer) **/

    /**
    * Struct of the mean_gradients_for_weights:
    * 
    *    mean_gradients_for_weights[ layer<LAYER_INDEX> ][ unit<UNIT_INDEX> ][ <WEIGHT_INDEX> ] = Number
    * 
    *    Or more visual explaination:
    * 
    *    mean_gradients_for_weights
    *     
    *       layer0
    *          --> unit0
    *              --> mean_of_accumulation_of_weight 1 
    *              --> mean_of_accumulation_of_weight 2
    *                  (etc.. other weights gradients) 
    * 
    *              (etc... other units)
    * 
    *      (etc... other layers)
    * 
    *            
    * So, the variable mean_gradients_for_weights Is a hashmap of the mean of the accumulated gradients for each weight( of each unit of each layer ) 
    * Is organized in this way!
    */
    let mean_gradients_for_weights = {}; //TODO RENOMEAR ISSO PRA mean_gradients_for_weights
    
    Object.keys(summed_gradients_for_weights).forEach(function( layerId ){
        let layerData  = summed_gradients_for_weights[ layerId ];
        let unitsId    = Object.keys(layerData);

        //If not existis the layer, create with empty object
        if( mean_gradients_for_weights[ layerId ] == undefined ){
            mean_gradients_for_weights[ layerId ] = {};
        }

        unitsId.forEach(function(unitId){
            //If not existis the unitID, create with empty object
            if( mean_gradients_for_weights[ layerId ][ unitId ] == undefined ){
                mean_gradients_for_weights[ layerId ][ unitId ] = [];
            }

            //Sum the unit gradient
            let unit_summed_gradient_for_weights = summed_gradients_for_weights[ layerId ][ unitId ];

            let number_of_weights = unit_summed_gradient_for_weights.length;

            for( let c = 0 ; c < number_of_weights ; c++ )
            {
                let weight_index = c;
                let sum_of_weight_c = unit_summed_gradient_for_weights[ weight_index ]; //Obtem o peso C que foi acumulado
                mean_gradients_for_weights[ layerId ][ unitId ][ weight_index ] = sum_of_weight_c / number_of_samples;
            }
        });
    });

    /**
    * Much similar to mean_gradients_for_weights
    * BUT, instead have a sub-object inside the unit object, they have just a value, that is the mean of the accumulated bias(a Number)
    *
    * Or more visual explaination:
    * 
    *    mean_gradients_for_bias
    *     
    *       layer0
    *          --> mean_of_accumulation_of_the_bias_of_unit0
    *          --> mean_of_accumulation_of_the_bias_of_unit1  
    *          --> mean_of_accumulation_of_the_bias_of_unit2        
    *              (etc... other bias means)
    * 
    *       (etc... other layers)
    * 
    * So, inside the mean_gradients_for_bias, he have layers, like in the other examples above
    * And each layer have N means of the accumulation of the bias of each unit( of each layer )
    */
    let mean_gradients_for_bias = {};
    
    Object.keys(summed_gradients_for_bias).forEach(function( layerId ){
        let layerData  = summed_gradients_for_bias[ layerId ];
        let unitsId    = Object.keys(layerData);

        //If not existis the layer, create with empty object
        if( mean_gradients_for_bias[ layerId ] == undefined ){
            mean_gradients_for_bias[ layerId ] = {};
        }

        unitsId.forEach(function(unitId){
            //If not existis the unitID, create with empty zero
            if( mean_gradients_for_bias[ layerId ][ unitId ] == undefined ){
                mean_gradients_for_bias[ layerId ][ unitId ] = 0;
            }

            //Sum the unit gradient
            let unit_summed_gradient_for_the_bias = summed_gradients_for_bias[ layerId ][ unitId ];
            mean_gradients_for_bias[ layerId ][ unitId ] = unit_summed_gradient_for_the_bias / number_of_samples;
        });
    });

    /**
    * Applies the Gradient Descent algorithm to update the parameters
    */
    modelContext.optimize_the_parameters( mean_gradients_for_weights, mean_gradients_for_bias );
}