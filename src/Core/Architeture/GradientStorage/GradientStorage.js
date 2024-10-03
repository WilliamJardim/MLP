//Namespace of the GradientStorage
net.GradientStorage = {};

//GradientStorage main
net.GradientStorage.GradientStorage = function(){
    let context = {};
    context.table = {};
    context.vincules = {};

    /**
    * Vinculate a prop to this object
    * @param {String}   newAttributeName
    * @param {any}      valueOfThisAttribute
    * @returns {Object} - Layer It self
    */
    context.vinculate = function(newAttributeName, valueOfThisAttribute){
        context[ newAttributeName ] = valueOfThisAttribute;
        context.vincules[ newAttributeName ] = context[ newAttributeName ];
        return context;
    }

    /**
    * Make all the gradients as a default value 
    */
    context.resetGradients = function(){
        context.table = {};
    }

    //The same as "context.resetGradients"
    context.startBlank = context.resetGradients;
    context.eraseAll   = context.resetGradients;

    /**
    * Create a instance to store the layer gradients 
    */
    context.registryLayer = function( layerNumber ){
        
        let layerGradientStorage = net.GradientStorage.LayerGradientStorage(function(storageContext){
            storageContext.vinculate('model_gradient_storage', context);
        });

        context.table[ `layer${ layerNumber }` ] = layerGradientStorage;
    }

    /**
    * Get the gradient table of a layer
    * @param {Number} layerNumber 
    * @returns {net.GradientStorage.LayerGradientsStorage}
    */
    context.getLayer = function( layerNumber ){
        return context.table[ `layer${ layerNumber }` ];
    }

    /**
    * Get the gradient table of a layer
    * @param {Number} layerNumber 
    * @returns {net.GradientStorage.LayerGradientsStorage}
    */
    context.ofLayer = function( layerNumber ){
        return context.getLayer( layerNumber );
    }

    /**
    * Alias for context.ofLayer
    * 
    * Get the gradient table of a layer
    * @param {Number} layerNumber 
    * @returns {net.GradientStorage.LayerGradientsStorage}
    */
    context.atLayer = context.ofLayer;

    /**
    * Set a gradient vector with respect to the parameters(weights and bias) of a especific unit of a especific layer
    * 
    * @param {Number} ofUnit                   - The unit in question 
    * @param {Number} ofLayer                  - The layer of the unit in question
    * @param {net.GradientVector} setGradient  - The gradient vector to be setted
    */
    context.setGradientWrtOf = function( { ofUnit, ofLayer, setGradient } ){

        context.atLayer( ofLayer )
               .atUnit( ofUnit )
               .setGradientWrtEstimative( setGradient );

    }

    return new Proxy(context, {
        get: function(target, prop, receiver) {
            // Se a propriedade solicitada for a instância (this), retorna 'table'
            if (prop === 'table') {
                return target.table;
            }

            // Permitir acesso direto ao 'table' por meio do proxy
            if (prop in target.table) {
                return target.table[prop];
            }

            // Retorna a própria propriedade, ou 'undefined' se não estiver presente
            return Reflect.get(target, prop, receiver);
        },

        set: function(target, prop, value) {
            // Define diretamente no 'table' se a propriedade existir
            if (prop in target.table || !isNaN(prop)) {
                target.table[prop] = value;
                return true;
            }

            // Define diretamente no objeto 'context' se não for numérico
            return Reflect.set(target, prop, value);
        }
    }); 
}