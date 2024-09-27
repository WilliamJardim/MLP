/**
* Multilayer Perceptron Neural Network (MLP)
* By William Alves Jardim
* 
* This implementation is entirely original, written from scratch in JavaScript.
* It was inspired by various publicly available resources, including concepts 
* and explanations from the work of Jason Brownlee on backpropagation.
* 
* CREDITS && REFERENCE:
* Jason Brownlee, How to Code a Neural Network with Backpropagation In Python (from scratch), Machine Learning Mastery, Available from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/, accessed April 15th, 2024.
* 
* For more details, see README.md.
*/

//A Unit(with just estimateValue and weight initialization)
net.Unit = function( unit_config={}, afterCreateCallback=()=>{}  ){
    let context = {};

    context.objectName            = 'Unit';
    context.afterCreateCallback   = afterCreateCallback;

    //Parameters that will be received in creation
    context._layerRef             = unit_config._layerRef;
    context._unitIndex            = unit_config._unitIndex;

    //External vincules
    context.model_parameters      = unit_config.model_parameters;

    //Parameters
    context.number_of_inputs      = unit_config.number_of_inputs     || Number();
    context.activation_function   = unit_config.activation_function  || 'sigmoid';
    //context.weights               = unit_config.weights              || Array(context.number_of_inputs).fill(0);
    //context.bias                  = unit_config.bias                 || Number();

    context.vincules              = {}; //All vincules

    /**
    * Get the layer object( That was linked to this unit )
    * @returns {Object}
    */
    context.getLayerOfThisUnit = function(){
        return context._layerRef; 
    }

    /**
    * Get the MLP model
    */
    context.getModel = function(){
        return context.getLayerOfThisUnit().getFather();
    }

    /**
    * Get the index of this unit in the owner-layer
    * @returns {Number} 
    */
    context.getThisUnitIndex = function(){
        return context._unitIndex;
    }

    /**
    * Vinculate a prop to this unit
    * @param {String} newAttributeName
    * @param {any}    valueOfThisAttribute
    */
    context.vinculate = function(newAttributeName, valueOfThisAttribute){
        context[ newAttributeName ] = valueOfThisAttribute;
        context.vincules[ newAttributeName ] = context[ newAttributeName ];
    }

    /**
    * Get the activation function name
    */
    context.getFunctionName = function(){
        return context.activation_function;
    }

    //Setters
    context.setWeights = function( newWeights=[] ){
        if( !(newWeights instanceof Array) ){
            throw Error(`the newWeights=${newWeights} is not a Array instance!`);
        }
        if( newWeights.length == 0 ){
            throw Error(`the newWeights is empty Array!`);
        }  

        let index_of_layer     = context.getLayerOfThisUnit().getLayerIndex();
                                        
        let index_of_this_unit = context.getThisUnitIndex();     

        context.getModel()
               .manipulateModelParameter({
                    ofLayer  : index_of_layer,
                    ofUnit   : index_of_this_unit
                })
                .setWeightsArray( newWeights );
    }

    /**
    * Set a especific weight
    * @param {Number} weight_index 
    * @param {Number} value 
    */
    context.setWeightOfIndex = function( weight_index, value ){

        let index_of_layer     = context.getLayerOfThisUnit().getLayerIndex();
                                        
        let index_of_this_unit = context.getThisUnitIndex();     

        context.getModel()
               .manipulateModelParameter({
                    ofLayer  : index_of_layer,
                    ofUnit   : index_of_this_unit
                })
                .setWeightOfIndex( weight_index, Number(value) );

    }

    //Getter for the context.weights
    context.getWeights = function(){
        
        let index_of_layer     = context.getLayerOfThisUnit().getLayerIndex();
        let index_of_this_unit = context.getThisUnitIndex();     

        return context.getModel()
                      .manipulateModelParameter({
                          ofLayer  : index_of_layer,
                          ofUnit   : index_of_this_unit
                       })
                       .getWeights();
    }

    context.getBias = function(){

        let index_of_layer     = context.getLayerOfThisUnit().getLayerIndex();
        let index_of_this_unit = context.getThisUnitIndex();     

        let manipulator = context.getModel()
                                 .manipulateModelParameter({
                                        ofLayer  : index_of_layer,
                                        ofUnit   : index_of_this_unit
                                  });
        
        return manipulator.getBias();

    }

    //Add a value to a especific weight
    context.addWeight = function( weight_index, number ){

        let index_of_layer     = context.getLayerOfThisUnit().getLayerIndex();
        let index_of_this_unit = context.getThisUnitIndex();     

        let manipulator = context.getModel()
                                 .manipulateModelParameter({
                                        ofLayer  : index_of_layer,
                                        ofUnit   : index_of_this_unit
                                  });
        
        manipulator.setWeightOfIndex( weight_index, manipulator.getWeightOfIndex( weight_index ) + number );

    }

    //Subtract a value to a especific weight
    context.subtractWeight = function( weight_index, number ){
        
        let index_of_layer     = context.getLayerOfThisUnit().getLayerIndex();
        let index_of_this_unit = context.getThisUnitIndex();     

        let manipulator = context.getModel()
                                 .manipulateModelParameter({
                                        ofLayer  : index_of_layer,
                                        ofUnit   : index_of_this_unit
                                  });
        
        manipulator.setWeightOfIndex( weight_index , manipulator.getWeightOfIndex( weight_index ) - number );

    }

    //Get a WeightManipulator for a especific weight
    context.selectWeight = function( weight_index ){

        return net.WeightManipulator({
            context: context,
            index: weight_index
        });

    }

    //Setter for bias
    context.setBias = function( newBias=Number() ){

        if( typeof newBias != 'number' ){
            throw Error(`the newBias=${newBias} is not a Number instance!`);
        }

        let index_of_layer     = context.getLayerOfThisUnit().getLayerIndex();
        let index_of_this_unit = context.getThisUnitIndex();     

        context.getModel()
               .manipulateModelParameter({
                    ofLayer  : index_of_layer,
                    ofUnit   : index_of_this_unit
                })
                .setBias( Number(newBias) );

    }

    //Add a value for the bias
    context.addBias = function( number ){

        let index_of_layer     = context.getLayerOfThisUnit().getLayerIndex();
        let index_of_this_unit = context.getThisUnitIndex();     

        let manipulator = context.getModel()
                                 .manipulateModelParameter({
                                        ofLayer  : index_of_layer,
                                        ofUnit   : index_of_this_unit
                                  });
        
        manipulator.setBias( manipulator.getBias() + number );
        
    }

    //Subtract a value for the bias
    context.subtractBias = function( number ){

        let index_of_layer     = context.getLayerOfThisUnit().getLayerIndex();
        let index_of_this_unit = context.getThisUnitIndex();     

        let manipulator = context.getModel()
                                 .manipulateModelParameter({
                                        ofLayer  : index_of_layer,
                                        ofUnit   : index_of_this_unit
                                  });
        
        manipulator.setBias( manipulator.getBias() - number );

    }

    /**
    * Generate random parameters
    */
    context.generate_random_parameters = function( number_of_inputs=context.number_of_inputs ){

        let parameterIndex    = 0;
        let masterRandomValue = Math.random();
        let totP = 0;

        while( parameterIndex < number_of_inputs )
        {
            let currentRandomValue = Math.random();
            
            context.setWeightOfIndex( parameterIndex, currentRandomValue );
            parameterIndex++;
            totP++;
        }
        
        context.setBias( masterRandomValue );
    }

    /**
    * Get a especific weight of this unit 
    * @param {Number} weight_index
    * @returns {Number}
    */
    context.getWeightOfIndex = function( weight_index ){

        let index_of_layer     = context.getLayerOfThisUnit().getLayerIndex();
        let index_of_this_unit = context.getThisUnitIndex();     

        let manipulator = context.getModel()
                                 .manipulateModelParameter({
                                        ofLayer  : index_of_layer,
                                        ofUnit   : index_of_this_unit
                                  });

        return manipulator.getWeightOfIndex( weight_index );
        
    }

    /**
    * Get a especific input of weight
    * @param {Number} weight_index 
    * @returns {Number}
    */
    context.getInputOfWeight = function( weight_index ){
        let father_layer          = context.getLayerOfThisUnit(), //Get the layer to which THIS UNIT belongs
            father_layer_inputs   = father_layer.getInputs();     //The inputs of the layer(that is, the layer to which THIS UNIT belongs)

        return father_layer_inputs[ weight_index ];
    }

    /**
    * Get the estimated value for ONE SAMPLE of this UNIT
    * @param   {Array}   sample_inputs        - The model inputs
    * @returns {Number}                       - The model estimative
    */
    context.estimateValue = function( sample_inputs )
    {
        let number_of_inputs = sample_inputs.length;
        let summed_value = 0;
        let i = 0;

        /**
        * Combine the inputs with their weights 
        */
        while( i < number_of_inputs  ) 
        {
            let weight_index = i;
            summed_value = summed_value + ( sample_inputs[i] * context.getWeightOfIndex( weight_index ) );

            //Next iteration
            i++;
        }

        /**
        * Add the bias
        * Relemering that the bias does have a input, then, i put 1, to ilustrate this in code
        */
        summed_value = summed_value + ( 1 * context.getBias() );

        let estimative = net.activations[ context.getFunctionName() ]( summed_value );

        return {
            activation_function_result: estimative,
            unit_potential: summed_value //The unit activation potential(just the summed value)
        };
    }

    /**
    * Get the own context
    * @returns {Object} - the layer it self
    */
    context.getSelfContext = function(){
        return context;
    }

    /**
    * Get the own context
    * @returns {Object} - the layer it self
    */
    context.atSelf = context.getSelfContext;

    //Run the callback
    context.afterCreateCallback.bind(context)( context );

    return context;
}