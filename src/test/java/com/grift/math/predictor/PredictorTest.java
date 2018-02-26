package com.grift.math.predictor;

import java.util.List;
import java.util.stream.IntStream;
import com.google.common.collect.Lists;
import com.grift.forex.symbol.ImmutableSymbolIndexMap;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.math.ProbabilityVector;
import com.grift.math.real.Real;
import org.jetbrains.annotations.NotNull;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;
import org.springframework.util.StopWatch;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

@RunWith(MockitoJUnitRunner.class)
public class PredictorTest {
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
        Real[] result = Predictor.projectR2(Real.valueOf(0.2), Real.valueOf(0.2));
        assertEquals("should be same vector", Real.valueOf(0.2), result[0]);
        assertEquals("should be same vector", Real.valueOf(0.8), result[1]);
    }

    @Test @Ignore
    public void ensureEquilibrium() {
        final int dimension = 300;
        final int iterations = 2000;
        final int[] misfireCount = {0};

        ImmutableSymbolIndexMap map = new ImmutableSymbolIndexMap(getStrings(dimension));
        ProbabilityVector.Factory factory = new ProbabilityVector.Factory(map);
        Predictor predictor = new Predictor(map);
        StopWatch watch = new StopWatch();
        IntStream.range(0, iterations).forEach(i -> {
            final ProbabilityVector[] v1 = {factory.create(randomVals(dimension))};
            final ProbabilityVector[] v2 = {wiggle(factory, v1[0])};
            watch.start();
            ProbabilityVector result = predictor.getPrediction(v1[0], v2[0]);
            watch.stop();
            for (double d : result.getValues()) {
                if (Double.isNaN(d) || Double.isInfinite(d) || d < 0 || d > 1) {
                    misfireCount[0]++;
                    break;
                }
            }
        });
        long totalTime = watch.getTotalTimeMillis();
        double averageMilis = ((double) totalTime) / ((double) iterations);
        double misfirePercentage = 100 * ((double) misfireCount[0]) / ((double) iterations);
    }

    private double[] randomVals(int dimension) {
        return IntStream.range(0, dimension).mapToDouble(i -> 100 * Math.random()).toArray();
    }

    @NotNull
    private String[] getStrings(int dimension) {
        List<String> arr = Lists.newArrayList();
        for (int i = 0; i < dimension; i++) {
            arr.add(String.format("%03d", i));
        }
        return arr.toArray(new String[dimension]);
    }

    private ProbabilityVector wiggle(ProbabilityVector.Factory factory, ProbabilityVector v1) {
        ProbabilityVector w = factory.copy(v1);
        double delta = Math.random() / 1000 * Math.pow(-1, Math.round(100 * Math.random()));
        double ratio = 1 + delta;
        int index = (int) (Math.round(100 * Math.random()) % v1.getDimension());
        w.put(index, v1.get(index) * ratio);
        return w;
    }
}