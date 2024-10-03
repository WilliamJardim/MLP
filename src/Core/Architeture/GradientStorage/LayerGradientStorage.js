net.GradientStorage.LayerGradientStorage = function(callback){
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
    * Get the "UnitGradientStorage" of a especific unit, that contains the GradientVector of this especific unit itself
    * @param {Number} unitNumber 
    * @returns {net.GradientStorage.UnitGradientStorage}
    */
    context.getUnit = function( unitNumber ){
        return context.table[ `unit${ unitNumber }` ];
    }

    /**
    * Return if a key to the especific unit parameters exists or not in this LayerGradientStorage instance
    * @param {Number} unitNumber 
    * @returns {Boolean}
    */
    context.unitExists = function( unitNumber ){
        return context.getUnit( unitNumber ) != undefined;
    }

    /**
    * Set a "UnitGradientStorage" to the especific unit paramteres in this LayerGradientStorage instance
    * @param {Number} unitNumber 
    * @returns {Boolean}
    */
    context.setUnit = function( unitNumber, obj ){
        context.table[ `unit${ unitNumber }` ] = obj;
    }

    /**
    * Get the gradient table of a unit
    *
    * if not exists will create a new UnitGradientStorage,
    * and if existis, will simply return the UnitGradientStorage reference
    * 
    * @param {Number} unitNumber 
    * @returns {net.GradientStorage.LayerGradientsStorage}
    */
    context.ofUnit = function( unitNumber ){
        if( context.unitExists( unitNumber ) == false ){
            context.setUnit( unitNumber, net.GradientStorage.UnitGradientStorage() );
            return context.getUnit( unitNumber );
        }
        return context.getUnit( unitNumber );
    }

    /**
    * Alias for context.ofUnit 
    * 
    * Get the gradient table of a unit
    *
    * if not exists will create a new UnitGradientStorage,
    * and if existis, will simply return the UnitGradientStorage reference
    * 
    * @param {Number} unitNumber 
    * @returns {net.GradientStorage.LayerGradientsStorage}
    */
    context.atUnit = context.ofUnit;

    callback(context);

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