/**
* A utility for any parameter in model

* @param {Object} model_parameters - The object that stores all the parameters of the MLP model
* @param {Number} ofUnit           - The unit in question
* @param {Number} ofLayer          - The layer of the unit in question
*/
net.ParameterManipulator = function( { model_parameters, ofUnit, ofLayer } ){
    let context               = {};
    context.model_parameters  = model_parameters;
    context.ofUnit            = ofUnit;
    context.ofLayer           = ofLayer;

    /**
    * Get layer parameter context 
    */
    context.getLayerParameterContext = function(){
       let layerID = `layer${ context.ofLayer }` ;
       
       if( context.model_parameters[ layerID ] == undefined ){
           context.model_parameters[ layerID ] = {};
       }

       return context.model_parameters[ layerID ];
    }

    /**
    * Get the parameters of the unit 'ofUnit'
    * @returns {Object}
    */
    context.getUnitParameterContext = function(){
        let unitID = `unit${ context.ofUnit }`;
        let layerParameters = context.getLayerParameterContext();

        if( layerParameters[ unitID ] == undefined ){
            layerParameters[ unitID ] = {};
            layerParameters[ unitID ].weights = [];
            layerParameters[ unitID ].bias = NaN;
        }

        return layerParameters[ unitID ];
    }

    context.setWeightOfIndex = function( weight_index, value ){
        context.getUnitParameterContext().weights[ weight_index ] = value;
    }

    context.getWeights = function(){
        return context.getUnitParameterContext().weights;
    }

    context.setWeightsArray = function( newWeights ){
        context.getUnitParameterContext().weights = newWeights;
    }

    context.getWeightOfIndex = function( weight_index ){
        return context.getUnitParameterContext().weights[ weight_index ];
    }

    context.setBias = function( value ){
        context.getUnitParameterContext().bias = value;
    }

    context.getBias = function(){
        return context.getUnitParameterContext().bias;
    }

    return context;
}