net.GradientStorage = function(){
    let context = {};
    context.table = {};

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