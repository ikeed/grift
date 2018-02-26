package com.grift.spring.controller;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import com.grift.GriftApplication;
import com.grift.forex.symbol.ImmutableSymbolIndexMap;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.forex.symbol.SymbolPair;
import com.grift.math.ProbabilityVector;
import com.grift.math.real.Real;
import com.grift.spring.service.DecoupleService;
import org.assertj.core.util.Lists;
import org.jetbrains.annotations.NotNull;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringRunner;

import static com.grift.math.real.Real.ZERO;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

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
        List<Map<SymbolPair, Real>> result = predictionController.getPrediction(Lists.emptyList());
        assertNotNull(result);
        assertEquals(0, result.size());
    }

    @Test
    public void getPredictionTrivial() {
        List<ProbabilityVector> vectors = Lists.newArrayList();
        ProbabilityVector vector = vectorFactory.create(0.2, 0.3, 0.5);
        vectors.add(vector);
        vectors.add(vectorFactory.copy(vector));
        Map<SymbolPair, Real> expectedValues = getRecoupledVector(vector);

        List<Map<SymbolPair, Real>> result = predictionController.getPrediction(vectors);

        assertNotNull(result);
        assertEquals(1, result.size());
        Map<SymbolPair, Real> mapResult = result.get(0);
        mapResult.forEach((key, value) -> assertTrue("Wrong value", expectedValues.get(key).equals(value)));
    }

    @NotNull
    private Map<SymbolPair, Real> getRecoupledVector(ProbabilityVector vector) {
        return decouplerService.recouple(symbolPairs, vector);
    }

    @Test
    public void getPredictionNonTrivial() {
        List<ProbabilityVector> vectors = Lists.newArrayList(
                vectorFactory.create(0.2, 0.3, 0.5),
                vectorFactory.create(0.19, 0.31, 0.5)
        );
        ProbabilityVector expectedVector = vectorFactory.create(0.1779809145894581071192365835783242847694633431329713907785337253188556311413490127542252456539605102, 0.3220190854105418728807634164216749152305366568669966092214662746798643688586509871945747543460394878, 0.5000000000000000200000000000000008000000000000000320000000000000012800000000000000512000000000000020);
        Map<SymbolPair, Real> expectedValues = getRecoupledVector(expectedVector);

        List<Map<SymbolPair, Real>> result = predictionController.getPrediction(vectors);

        assertNotNull(result);
        assertEquals(1, result.size());
        Map<SymbolPair, Real> mapResult = result.get(0);
        Real dist = getVectorDistance(mapResult, expectedValues);
        assertTrue(dist.isZero());
    }

    private Real getVectorDistance(Map<SymbolPair,Real> result, Map<SymbolPair, Real> expectedValues) {
        Real total = ZERO;

        for (SymbolPair key : result.keySet()) {
            Real diff = result.get(key).subtract(expectedValues.get(key));
            total = total.plus(diff.pow(2));
        }
        return total.sqrt();
    }
}