net.GradientStorage.UnitGradientStorage = function(){
    let context = {};
    context.gradientVector = {}; //A instance of the net.GradientVector, but in the first time is a empty object that will be attribuited
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
    * Set the gradient vector with respect the parameters of a especific unit
    * @param {net.GradientVector} gradientVector 
    */
    context.setGradientWrtEstimative = function( gradientVector ){
        context.gradientVector = gradientVector;
    }

    return new Proxy(context, {
        get: function(target, prop, receiver) {
            // Se a propriedade solicitada for a instância (this), retorna 'gradientVector'
            if (prop === 'table') {
                return target.gradientVector;
            }

            // Permitir acesso direto ao 'gradientVector' por meio do proxy
            if (prop in target.gradientVector) {
                return target.gradientVector[prop];
            }

            // Retorna a própria propriedade, ou 'undefined' se não estiver presente
            return Reflect.get(target, prop, receiver);
        },

        set: function(target, prop, value) {
            // Define diretamente no 'gradientVector' se a propriedade existir
            if (prop in target.gradientVector || !isNaN(prop)) {
                target.gradientVector[prop] = value;
                return true;
            }

            // Define diretamente no objeto 'context' se não for numérico
            return Reflect.set(target, prop, value);
        }
    }); 
}