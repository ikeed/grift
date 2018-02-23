package com.grift.math.predictor;

import com.google.common.collect.Lists;
import com.grift.forex.symbol.ImmutableSymbolIndexMap;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.math.ProbabilityVector;
import org.jetbrains.annotations.NotNull;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

@RunWith(MockitoJUnitRunner.class)
public class PredictorTest {

    public static final double DELTA = 0.0000001;
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

    @Test(expected = IllegalArgumentException.class)
    public void differingDimensions() {
        ProbabilityVector v1 = getProbabilityVector(10, 20, 30);
        ProbabilityVector v2 = new ProbabilityVector(new SymbolIndexMap().addSymbols(Lists.newArrayList("CAD", "USD")).getImmutablecopy(), new double[]{10, 20});
        new Predictor(v1, v2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void negativeElement() {
        ProbabilityVector v1 = getProbabilityVector(-10, -20, -30);
        ProbabilityVector v2 = getProbabilityVector(10, -20, 30);
        new Predictor(v1, v2);
    }

    @Test
    public void sameVector() {
        ProbabilityVector v1 = getProbabilityVector(10, 20, 30);
        ProbabilityVector v2 = new ProbabilityVector(symbolIndexMap, v1.getValues());

        Predictor predictor = new Predictor(v1, v2);

        ProbabilityVector result = predictor.getPrediction();
        assertNotNull(result);
        assertTrue("result should match input", v1.equals(result));
    }

    @Test
    public void specificTest() {
        symbolIndexMap = new SymbolIndexMap().addSymbols(Lists.newArrayList("CAD", "USD")).getImmutablecopy();
        ProbabilityVector v1 = getProbabilityVector(0.2, 0.8);
        ProbabilityVector v2 = getProbabilityVector(0.5, 0.5);
        double firstComponent = 0.13798;
        double secondComponent = 0.11202;
        double sum = firstComponent + secondComponent;
        firstComponent /= sum;
        secondComponent /= sum;

        assertEquals("sanity check", firstComponent, 1 - secondComponent, 0.0000001);


        double[] result = new Predictor(v1, v2).getPrediction().getValues();

        assertEquals("differed in first component", firstComponent, result[0], 0.00001);
        assertEquals("differed in second component", secondComponent, result[1], 0.00001);
    }

    @Test
    public void projectR2SameVector() {
        double[] o = new double[] { 0.2, 0.8};
        double[] n = new double[] { 0.2, 0.8};
        double[] result = Predictor.project_R2(o, n);
        assertEquals("should be same vector", o[0], result[0], DELTA);
        assertEquals("should be same vector", o[1], result[1], DELTA);
    }

    @NotNull
    private ProbabilityVector getProbabilityVector(@NotNull double... vals) {
        return new ProbabilityVector(symbolIndexMap, vals);
    }
}