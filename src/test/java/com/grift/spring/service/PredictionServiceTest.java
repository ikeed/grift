package com.grift.spring.service;

import com.google.common.collect.Lists;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.math.ProbabilityVector;
import com.grift.math.predictor.Predictor;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class PredictionServiceTest {

    private Predictor predictor;
    private PredictionService predictionService;
    private SymbolIndexMap symbolIndexMap;
    private ProbabilityVector.Factory vectorFactory;

    @Before
    public void setup() {
        symbolIndexMap = new SymbolIndexMap()
                .addSymbols(Lists.newArrayList("CAD", "USD", "GBP"));
        predictor = new Predictor(symbolIndexMap.getImmutableCopy());
        predictionService = new PredictionService(predictor);
        vectorFactory = new ProbabilityVector.Factory(symbolIndexMap.getImmutableCopy());
    }

    @Test
    public void makeTrivialPrediction() {
        ProbabilityVector v = vectorFactory.create(0.2, 0.2, 0.6);
        ProbabilityVector result = predictionService.makePrediction(v, vectorFactory.copy(v));
        assertEquals("<0.2, 0.2, 0.6>", result.toString());
    }
}