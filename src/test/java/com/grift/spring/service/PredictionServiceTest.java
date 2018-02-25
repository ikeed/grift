package com.grift.spring.service;

import java.util.ArrayList;
import com.google.common.collect.Lists;
import com.grift.forex.symbol.ImmutableSymbolIndexMap;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.math.ProbabilityVector;
import com.grift.math.predictor.Predictor;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class PredictionServiceTest {

    private PredictionService predictionService;
    private ProbabilityVector.Factory vectorFactory;

    @Before
    public void setup() {
        ArrayList<String> symbols = Lists.newArrayList("CAD", "USD", "GBP");
        ImmutableSymbolIndexMap immutableSymbolIndexMap = new SymbolIndexMap().addSymbols(symbols).getImmutableCopy();
        Predictor predictor = new Predictor(immutableSymbolIndexMap);
        predictionService = new PredictionService(predictor);
        vectorFactory = new ProbabilityVector.Factory(immutableSymbolIndexMap);
    }

    @Test
    public void makeTrivialPrediction() {
        ProbabilityVector v = vectorFactory.create(0.2, 0.2, 0.6);
        ProbabilityVector result = predictionService.makePrediction(v, vectorFactory.copy(v));
        assertEquals("<0.2, 0.2, 0.6>", result.toString());
    }

    @Test
    public void makeNontrivialPrediction() {
        ProbabilityVector oldVector = vectorFactory.create(0.2, 0.2, 0.6);
        ProbabilityVector newVector = vectorFactory.create(0.25, 0.18, 0.57);
        ProbabilityVector result = predictionService.makePrediction(oldVector, newVector);
        assertEquals("<0.286448067655979, 0.1772020329361286, 0.5363498994078925>", result.toString());
    }
}