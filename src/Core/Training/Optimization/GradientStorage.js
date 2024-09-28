/**
* A object that represents the gradient of weight and bias OF A UNIT
*
* In the mathematical, the gradient is a vector, where each index is the derivative with respect to a parameter
* So, thinking in the "with respect weights", if we have 3 weights, the gradient will be a vector with 3 elements: derivative with respect to first weight, derivative with respect to secound weight, and finnaly derivative with respect to threedy weight
* And the gradient with respect to the bias will be just the derivative with respect to bias
*
* @param {Array} weights - Derivative with respect to the weights(calculated gradient)
* @param {Number} bias   - Derivative with respect to the bias(calculated gradient)
* @returns {Object}
*/
net.MLP.prototype.Gradient = function( { weights, bias } ){
    let context = {};
    context.objectName = 'Gradient';
    
    context.weights     = context.weights || {}; //Contains the partial derivatives with respect to each weight
    context.bias        = context.bias    || {}; //Contain the partial derivative with respect to the bias

    context.getGradientsOf = function( targetObject ){

        let retrived = {};

        switch(targetObject){
            case 'weights':
                retrived = context.weights;
                break;

            case 'bias':
                retrived = context.bias;
                break;
        }

        return retrived;

    }

    context.setBiasDerivative = function( biasDerivative ){
        context.bias = biasDerivative;
    }

    context.setWeightDerivative = function( weightIndex, weightDerivative ){
        context.weights[ weightIndex ] = weightDerivative;
    }

    /**
    * Get the partial derivative with respect to a especific weight of the unit
    * @param {Number} weightIndex 
    * @returns {Number}
    */
    context.getDerivativeOfWeight = function( weightIndex ){
        return context.weights[ weightIndex ];
    }

    /**
    * Get the partial derivative with respect to the bias of the unit
    * @returns {Number}
    */
    context.getDerivativeOfBias = function(){
        return context.bias;
    }

    /**
    * The metheod "getMasterDerivativeOfTheUnit" allwo to obtaining the derivative of the bias because the derivative of the bias has no entry, 
    * 
    * (Why are we obtaining the derivative with respect to the "this hidden unit estimative function", and not with respect to the parameters on which "this hidden unit estimative function" depends.)
    * And what we want is precisely the derivative with respect to the result of the this hidden unit estimation function.  
    * 
    * In other words, here we are obtaining the derivative of the "neural network's estimation function result" with respect to the result of the "current unit function result"
    *
    * To facilitate understanding, you can think that the derivative of the "neural network's estimation function" with respect to the bias of a unit in a hidden layer IS THE SAME THING as the master derivative of that unit, so to speak.
    *
    * So every time you see me referring to "the derivative with respect to the bias of the Unit", you already know that this is the derivative of the "neural network estimative" with respect to the "unit estimative" of the layer.
    */
    context.getMasterDerivativeOfTheUnit = function(){
        return context.bias;
    }

    return context;
}

/**
* Store the "net.Gradient" of each unit of a especific layer
* NOTE: Each layer will have a "GradientsTable"
*/
net.MLP.prototype.GradientsTable = function(){
    let context = {};
    context.gradients = {};
    context.objectName = 'GradientsTable';

    /** Get the Gradient of a unit  */
    context.getGradientOfAUnit = function( unitIndex ){

        //If the gradient of the unit not exists
        if( context.gradients[ `unit${unitIndex}` ] == undefined ){

            //Creating a new empty Gradient because is the first time
            context.gradients[ `unit${unitIndex}` ] = net.MLP.prototype.Gradient({ weights: {}, bias: {} }); 

        }

        return context.gradients[ `unit${unitIndex}` ];
    }

    /**
    * Set the derivatie of especific parameters(that can be the Bias or Weights)
    * @param {String} the      - The parameter type: bias or weight
    * @param {Number} wIndex   - (If is a weight) the index of the weight 
    * @param {Number} ofUnit   - The index of the unit in question
    * @param {Number} beEqual  - The value to set as derivative of the parameter
    */
    context.setDerivativeOf = function( {the, wIndex, ofUnit, beEqual} ){
        
        switch( the ){
            case 'bias':
                context.getGradientOfAUnit( ofUnit )
                       .setBiasDerivative( beEqual );
                break;
            
            case 'weight':
                context.getGradientOfAUnit( ofUnit )
                       .setWeightDerivative( wIndex, beEqual );
                break;
        }

    }

    return new Proxy(context, {
        get: function(target, prop, receiver) {

          if (typeof prop === 'string' && !isNaN(prop) ) {

              if( Number(prop) === 0 ){
                 return target.gradients;

              }else{
                 throw Error(`Invalid prop!`);
              }
    
          }
          
          return Reflect.get(target, prop, receiver);
        },

        set: function(target, prop, value) {
          return Reflect.set(target, prop, value);
        }
    });
}

/**
* A utility for store gradients
* @param {Object} initialContent - The layer of the unit in question
*/
net.MLP.prototype.GradientStorage = function( { modelRef } ){
    let context = {};
    context.objectName = 'GradientStorage';
    
    context.content = {};
    context._modelContext = modelRef;

    /**
    * Get the Gradients Table of a especific layer
    * @param {*} layerIndex 
    * @returns 
    */
    context.getTableOfLayer = function( layerIndex ){
        return context.content[ `layer${ layerIndex }` ];
    }

    /**
    * Create a slot to store the gradients of especific a layer
    * @param {Number} layerIndex 
    */
    context.registryLayer = function( layerIndex ){

        //Registry a table to store the gradients of the especific layer
        context.content[ `layer${ layerIndex }` ] = net.MLP.prototype.GradientsTable();

        return context.content[ `layer${ layerIndex }` ]; //Return the created slot
    }

    /**
    * Compute and store the gradients of the parameters of this unit
    * @param {Number} derivative - The derivative
    * @param {Object} ofUnit     - The unit in question
    * @param {Object} ofLayer    - The layer of the unit in question
    */
    context.storeGradientOf = function( {derivative, ofUnit, ofLayer} ){

        let derivative_of_unit_result = derivative; //That is, the derivative with respect the unit function result

        /** 
        * Calculate the derivative of the bias
        * Do this setting the "derivative_of_unit_result" argument AS BE EQUAL the value of the derivative of bias of the unit(in the gradients map)
        * In other words, we are saying that the derivative of the bias of the unit WILL BE equal(=) "derivative_of_unit_result"
        * That is, the derivative of the LOSS of "estimative function" with respect to the bias
        */
        context.getTableOfLayer(ofLayer)
               .setDerivativeOf( { the      : 'bias', 
                                   ofUnit   : ofUnit, 
                                   beEqual  : derivative_of_unit_result } );

        /** Calculate the gradients of each weight of the unit */
        let unitWeights = modelRef.getWeightsOf( { ofUnit: ofUnit, 
                                                   ofLayer: ofLayer } );

        let currentWeightIndex = 0;
        while( currentWeightIndex < unitWeights.length )
        {
            /** Get the input of the current weight */
            let weight_input = modelRef.getLayer( ofLayer )
                                       .getUnit( ofUnit )
                                       .getInputOfWeight( currentWeightIndex );  

            /** Calculate the derivative with respect to the current weight  */
            let weight_derivative = weight_input * derivative_of_unit_result;

            /** 
            * Calculate the derivative of the current weight of the unit
            * Do this setting the "weight_derivative" AS BE EQUAL the derivative of the current weight of the unit
            * In other words, we are saying that the derivative of the current weight of the unit WILL BE equal(=) "weight_derivative"
            * That is, the derivative of the LOSS of "estimative function" with respect to the current weight
            */
            context.getTableOfLayer(ofLayer)
                   .setDerivativeOf( { the      : 'weight', 
                                       wIndex   : currentWeightIndex, 
                                       ofUnit   : ofUnit,
                                       beEqual  : weight_derivative } );

            //Next weight
            currentWeightIndex++;
        }
    }

    /**
    * Get the gradients of a especific layer 
    */
    context.retriveGradientsOfLayer = function( layerIndex ){
        return context.getTableOfLayer(layerIndex);
    }

    context.getGradients = function(){
        return context.content;
    }

    return context;
}