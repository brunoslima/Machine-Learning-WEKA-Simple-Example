/*
 * Problema de classificação
 * Precisa de uma base histórica para fazer a previsão
 * Determinar se uma pessoa vai gasar muito ou não
 */
package testeweka;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Bruno S Lima
 */
public class WekaExample1 {


    public static void main(String[] args) throws Exception{
        
        //Importar o data source
        DataSource ds = new DataSource("src/testeweka/vendas.arff");
    
        //Coloca todas as instancias do data souce em uma unico objeto
        Instances ins = ds.getDataSet();
        //Mostrar a base de dados
        //System.out.println(ins.toString());
        
        //Objetivo = PREVER SE UM DETERMINADO PERFIL DE USUÁRIO É MUITO GASTADOR OU NÃO
        ins.setClassIndex(3); //Indicando para o Weka qual é o atributo que eu desejo fazer a previsão, no caso (3 - gasta_muito)
        
        //Instanciando o algoritmo classificador (NB)
        NaiveBayes nb = new NaiveBayes();
        //Construindo o classificador
        nb.buildClassifier(ins);
        
        //Criando uma nova instancia passando o numero total de atributos dos meus dados (4)
        //Essa nova instancia em nosso exemplo corresponde o novo cliente que será classificado como gasta muito ou não.
        Instance newins = new DenseInstance(4);
        newins.setDataset(ins); //Relacionando a nova instancia a base de dados (não está adicionando, apenas relacionando)
       
        //Setando os dados, perfil do usuário que deseja-se fazer uma previsão.
        String sexo = "F";
        String idadeFaixa = ">=40";
        String possuiFilhos = "Sim";
        
        newins.setValue(0,sexo);
        newins.setValue(1,idadeFaixa);
        newins.setValue(2,possuiFilhos);
        
        //O algoritmo de classificação vai dizer qual é a probabilidade dessa pessoa gastar muito ou pouco
        double probabilidade[] = nb.distributionForInstance(newins);
        //O parametro do vetor de probabilidade é os valores da classificação, 1 - sim, 0 - não, esses valores são definidos no conjunto de dados
        System.out.println("Sim: " + probabilidade[1]);
        System.out.println("Não: " + probabilidade[0]);
        
    }
    
}
