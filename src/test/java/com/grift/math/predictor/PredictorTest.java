package com.grift.math.predictor;

import com.google.common.collect.Lists;
import com.grift.forex.symbol.ImmutableSymbolIndexMap;
import com.grift.forex.symbol.ProbabilityVector;
import com.grift.forex.symbol.SymbolIndexMap;
import org.jetbrains.annotations.NotNull;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertNotNull;

@RunWith(MockitoJUnitRunner.class)
public class PredictorTest {

    private ImmutableSymbolIndexMap symbolIndexMap;

    @Before
    public void setup() {
        symbolIndexMap = new SymbolIndexMap().addSymbols(Lists.newArrayList("CAD", "USD", "GBP")).getImmutablecopy();
    }

    @Test
    public void getPrediction() {
        ProbabilityVector v1 = getProbabilityVector(10, 20, 30);
        ProbabilityVector v2 = getProbabilityVector(10, 20, 31);
        Predictor predictor = new Predictor(v1, v2);

        ProbabilityVector result = predictor.getPrediction();
        assertNotNull(result);
    }

    @NotNull
    private ProbabilityVector getProbabilityVector(double... vals) {
        return new ProbabilityVector(symbolIndexMap, vals);
    }
}