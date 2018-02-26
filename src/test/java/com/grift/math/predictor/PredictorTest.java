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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;
import org.springframework.util.StopWatch;

import static com.grift.math.real.Real.ONE;
import static org.hamcrest.MatcherAssert.assertThat;
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
        Real firstComponent = new Real(0.551920896510422490717978129114270296037866572864989839524244282486232889785392291006230138070421776);
        Real secondComponent = new Real(0.448079103489577509282021870885729703962133427135010160475755717513767110214607708993769861929578224);

        assertTrue("sanity check", firstComponent.equals(ONE.subtract(secondComponent)));

        Real[] result = predictor.getPrediction(v1, v2).getValues();

        assertTrue("differed in first component", firstComponent.setDigitsPrecision(2).equals(result[0].setDigitsPrecision(2)));
        assertTrue("differed in second component", secondComponent.setDigitsPrecision(2).equals(result[1].setDigitsPrecision(2)));
    }

    @Test
    public void projectR2SameVector() {
        Real[] result = Predictor.projectR2(Real.valueOf(0.2), Real.valueOf(0.2));
        assertEquals("should be same vector", Real.valueOf(0.2).setDigitsPrecision(2), result[0].setDigitsPrecision(2));
        assertEquals("should be same vector", Real.valueOf(0.8).setDigitsPrecision(2), result[1].setDigitsPrecision(2));
    }

    @Test
    public void ensureEquilibrium() {
        final int dimension = 300;
        final int iterations = 200;
        final int[] missed = {0};

        ImmutableSymbolIndexMap map = new ImmutableSymbolIndexMap(getStrings(dimension));
        ProbabilityVector.Factory factory = new ProbabilityVector.Factory(map);
        Predictor predictor = new Predictor(map);
        StopWatch watch = new StopWatch();
        IntStream.range(0, iterations).forEach(i -> {
            final ProbabilityVector v1 = factory.create(randomVals(dimension));
            final ProbabilityVector v2 = wiggle(factory, v1);
            watch.start();
            ProbabilityVector result = predictor.getPrediction(v1, v2);
            watch.stop();
            if (!checkResult(result)) {
                missed[0]++;
            }
        });
        long totalTime = watch.getTotalTimeMillis();
        double averageMilis = ((double) totalTime) / ((double) iterations);
        double missPercentage = 100 * ((double) missed[0]) / iterations;
        assertThat("taking too long", averageMilis < 75);
        assertEquals("Empty components", 0, missed[0]);
    }

    private boolean checkResult(ProbabilityVector result) {
        for (int i = 0; i < result.getDimension(); i++) {
            Real future = result.get(i);
            if (!future.isPositive()) {
                return false;
            }
        }
        return true;
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
        w.put(index, v1.get(index).times(Real.valueOf(ratio)));
        return w;
    }
}