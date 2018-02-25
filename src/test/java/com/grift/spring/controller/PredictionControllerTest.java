package com.grift.spring.controller;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import com.grift.GriftApplication;
import com.grift.forex.symbol.ImmutableSymbolIndexMap;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.forex.symbol.SymbolPair;
import com.grift.math.ProbabilityVector;
import com.grift.spring.service.DecoupleService;
import org.assertj.core.util.Lists;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringRunner;

import static org.apache.commons.math.util.MathUtils.EPSILON;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

@RunWith(SpringRunner.class)
@SpringBootTest(classes = {PredictionController.class})
@ContextConfiguration(name = "fixture", classes = {GriftApplication.class})
public class PredictionControllerTest {

    @Autowired
    private PredictionController predictionController;
    @Autowired
    private DecoupleService decouplerService;
    @Autowired
    private List<SymbolPair> symbolPairs;

    private ProbabilityVector.Factory vectorFactory;

    @Before
    public void setup() {
        ArrayList<String> symbols = Lists.newArrayList("USD", "CAD", "GBP");
        ImmutableSymbolIndexMap symbolMap = new SymbolIndexMap().addSymbols(symbols).getImmutableCopy();
        vectorFactory = new ProbabilityVector.Factory(symbolMap);
    }

    @Test
    public void getPredictionEmpty() {
        List<Map<SymbolPair, Double>> result = predictionController.getPrediction(Lists.emptyList());
        assertNotNull(result);
        assertEquals(0, result.size());
    }

    @Test
    public void getPredictionTrivial() {
        List<ProbabilityVector> vectors = Lists.newArrayList();
        ProbabilityVector vector = vectorFactory.create(0.2, 0.3, 0.5);
        vectors.add(vector);
        vectors.add(vectorFactory.copy(vector));
        Map<SymbolPair, Double> expectedValues = decouplerService.recouple(symbolPairs, vector);

        List<Map<SymbolPair, Double>> result = predictionController.getPrediction(vectors);

        assertNotNull(result);
        assertEquals(1, result.size());
        Map<SymbolPair, Double> mapResult = result.get(0);
        mapResult.forEach((key, value) -> assertEquals("Wrong value", expectedValues.get(key), value, EPSILON));
    }

    @Test
    public void getPredictionNonTrivial() {
        List<ProbabilityVector> vectors = Lists.newArrayList(
                vectorFactory.create(0.2, 0.3, 0.5),
                vectorFactory.create(0.19, 0.31, 0.5)
        );
        ProbabilityVector expectedVector = vectorFactory.create(0.1779809145894581, 0.32201908541054186, 0.5);
        Map<SymbolPair, Double> expectedValues = decouplerService.recouple(symbolPairs, expectedVector);

        List<Map<SymbolPair, Double>> result = predictionController.getPrediction(vectors);

        assertNotNull(result);
        assertEquals(1, result.size());
        Map<SymbolPair, Double> mapResult = result.get(0);
        mapResult.forEach((key, value) -> assertEquals("Wrong value", expectedValues.get(key), value, EPSILON));
    }
}