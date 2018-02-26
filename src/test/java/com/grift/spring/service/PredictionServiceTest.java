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
        assertEquals("[0.2, 0.2, 0.6]", result.toString());
    }

    @Test
    public void makeNontrivialPrediction() {
        ProbabilityVector oldVector = vectorFactory.create(0.2, 0.2, 0.6);
        ProbabilityVector newVector = vectorFactory.create(0.25, 0.18, 0.57);
        ProbabilityVector result = predictionService.makePrediction(oldVector, newVector);
        assertEquals("[0.2864480676559789106702591956396383098209492235314143309376135154928875448431781579843070756414535354, 0.1772020329361285587584612317667181622667046463784506358581641437641871195589471774646192683228372473, 0.5363498994078925305712795725936435279123461300901350332042223407429253355978746645510736560357092173]", result.toString());
    }
}