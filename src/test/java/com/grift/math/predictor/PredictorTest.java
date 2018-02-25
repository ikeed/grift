package com.grift.math.predictor;

import com.google.common.collect.Lists;
import com.grift.forex.symbol.ImmutableSymbolIndexMap;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.math.ProbabilityVector;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

@RunWith(MockitoJUnitRunner.class)
public class PredictorTest {

    private static final double DELTA = 0.0000001;

    private ImmutableSymbolIndexMap immutableSymbolIndexMap;
    private Predictor predictor;
    private ProbabilityVector.Factory vectorFactory;

    @Before
    public void setup() {
        immutableSymbolIndexMap = new ImmutableSymbolIndexMap("CAD", "USD", "GBP");
        predictor = new Predictor(immutableSymbolIndexMap);
        vectorFactory = new ProbabilityVector.Factory(immutableSymbolIndexMap);
    }

    @Test
    public void getPrediction() {
        ProbabilityVector oldVector = vectorFactory.create(10, 20, 30);
        ProbabilityVector newVector = vectorFactory.create(10, 20, 31);

        ProbabilityVector result = predictor.getPrediction(oldVector, newVector);
        assertNotNull(result);
    }

    @Test(expected = IllegalArgumentException.class)
    public void differingDimensions() {
        ProbabilityVector vector3d = vectorFactory.create(10, 20, 30);
        ImmutableSymbolIndexMap immutableSymbolIndexMap2d = new SymbolIndexMap().addSymbols(Lists.newArrayList("CAD", "USD")).getImmutableCopy();
        ProbabilityVector.Factory factory2d = new ProbabilityVector.Factory(immutableSymbolIndexMap2d);

        ProbabilityVector vector2d = factory2d.create(10, 20);
        new Predictor(immutableSymbolIndexMap).getPrediction(vector3d, vector2d);
    }

    @Test(expected = IllegalArgumentException.class)
    public void negativeElement() {
        ProbabilityVector v1 = vectorFactory.create(-10, -20, -30);
        ProbabilityVector v2 = vectorFactory.create(10, -20, 30);
        predictor.getPrediction(v1, v2);
    }

    @Test
    public void sameVector() {
        ProbabilityVector v1 = vectorFactory.create(10, 20, 30);
        ProbabilityVector v2 = vectorFactory.create(v1.getValues());

        ProbabilityVector result = predictor.getPrediction(v1, v2);

        assertNotNull(result);
        assertTrue("result should match input", v1.equals(result));
    }

    @Test
    public void specificTest() {
        ImmutableSymbolIndexMap map = new SymbolIndexMap().addSymbols(Lists.newArrayList("CAD", "USD")).getImmutableCopy();
        ProbabilityVector.Factory vectorFactory2d = new ProbabilityVector.Factory(map);
        ProbabilityVector v1 = vectorFactory2d.create(0.2, 0.8);
        ProbabilityVector v2 = vectorFactory2d.create(0.5, 0.5);
        Predictor predictor = new Predictor(map);
        double firstComponent = 0.13798;
        double secondComponent = 0.11202;
        double sum = firstComponent + secondComponent;
        firstComponent /= sum;
        secondComponent /= sum;

        assertEquals("sanity check", firstComponent, 1 - secondComponent, 0.0000001);


        double[] result = predictor.getPrediction(v1, v2).getValues();

        assertEquals("differed in first component", firstComponent, result[0], 0.00001);
        assertEquals("differed in second component", secondComponent, result[1], 0.00001);
    }

    @Test
    public void projectR2SameVector() {
        double[] o = new double[]{0.2, 0.8};
        double[] n = new double[]{0.2, 0.8};
        double[] result = Predictor.projectR2(o, n);
        assertEquals("should be same vector", o[0], result[0], DELTA);
        assertEquals("should be same vector", o[1], result[1], DELTA);
    }
}