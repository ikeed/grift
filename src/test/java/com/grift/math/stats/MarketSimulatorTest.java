package com.grift.math.stats;

import java.io.IOException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import com.google.common.collect.Lists;
import com.grift.GriftApplication;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.forex.symbol.SymbolPair;
import com.grift.math.real.Real;
import com.grift.model.Tick;
import com.grift.spring.service.DecoupleService;
import com.grift.spring.service.PredictionService;
import org.jetbrains.annotations.NotNull;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringRunner;

import static org.junit.Assert.assertFalse;

@Ignore
@RunWith(SpringRunner.class)
@ContextConfiguration(name = "fixture", classes = {GriftApplication.class})
class MarketSimulatorTest {

    @Autowired
    private DecoupleService decoupleService;

    @Autowired
    private List<SymbolPair> symbolList;

    @Autowired
    private SymbolIndexMap symbolIndexMap;

    @Autowired
    private PredictionService predictionService;
    private ProbabilityVector.Factory factory;

    @Before
    public void setup() {
        factory = new ProbabilityVector.Factory(symbolIndexMap.keySet().toArray(new String[symbolIndexMap.size()]));
    }

    @Test
    public void runMarketSim() throws Exception {
        MarketSimulator marketSimulator = new MarketSimulator(symbolList, decoupleService, factory.create(10, 25, 40), 200);
        List<Result> results = getResults(marketSimulator);

        testChart(results);
        assertFalse(results.isEmpty());
    }

    private void testChart(List<Result> results) throws IOException {
        Map<SymbolPair, double[]> tickMap = new HashMap<>();
        Map<SymbolPair, double[]> predictionMap = new HashMap<>();

        for (SymbolPair pair : symbolList) {
            tickMap.put(pair, new double[results.size()]);
            predictionMap.put(pair, new double[results.size()]);
        }

        for (int i = 0; i < results.size(); i++) {
            Result result = results.get(i);
            for (SymbolPair pair : result.recoupledPrediction.keySet()) {
                tickMap.get(pair)[i] = result.tick.get(pair).toDouble();
                predictionMap.get(pair)[i] = getSmoothedPrediction(results, pair, i);
            }
        }

        List<double[]> arrays = new ArrayList<>(tickMap.values());
        arrays.addAll(predictionMap.values());
        doMaxMinNormalization(arrays);

        XYChart chart = getChart(symbolList, tickMap, predictionMap);
        new SwingWrapper<>(chart).displayChart();
        // Save it
        BitmapEncoder.saveBitmap(chart, "./Sample_Chart", BitmapEncoder.BitmapFormat.PNG);

        // or save it in high-res
        BitmapEncoder.saveBitmapWithDPI(chart, "./Sample_Chart_300_DPI", BitmapEncoder.BitmapFormat.PNG, 300);
    }

    private void doMaxMinNormalization(List<double[]> arrays) {
        arrays.forEach(this::doMaxMinNormalization);
    }

    private void doMaxMinNormalization(double[] arr) {
        double max = Arrays.stream(arr).max().getAsDouble();
        double min = Arrays.stream(arr).min().getAsDouble();
        for (int i = 0; i < arr.length; i++) {
            arr[i] = (arr[i] - min) / (max - min);
        }
    }

    private double getSmoothedPrediction(List<Result> results, SymbolPair pair, int centre) {
        double num = 0;
        double den = 0;
        double base = 0.8;
        int halfShell = 6;

        for (int i = Math.max(0, centre - halfShell); i <= Math.min(centre + halfShell, results.size() - 1); i++) {
            Result result = results.get(i);
            double weight = Math.pow(base, Math.abs(centre - i));
            num += result.recoupledPrediction.get(pair).toDouble() * weight;
            den += weight;
        }
        return num / den;
    }

    private XYChart getChart(List<SymbolPair> symbolList, Map<SymbolPair, double[]> tickMap, Map<SymbolPair, double[]> predictionMap) {

        // Create Chart
        XYChart chart = new XYChartBuilder().title("Currency Analysis").height(600).width(800).build();

        // Customize Chart

        for (SymbolPair pair : symbolList) {
            chart.addSeries(pair.toString(), null, tickMap.get(pair));
            chart.addSeries(pair.toString() + " Prediction", null, predictionMap.get(pair));
        }
        return chart;
    }

    @NotNull
    private List<Result> getResults(MarketSimulator marketSimulator) {
        List<Result> results = Lists.newArrayList();
        final ProbabilityVector[] prev = {null};
        marketSimulator.forEachRemaining(map -> {
            for (ProbabilityVector vec : insertTicksFromStateMap(map)) {
                if (prev[0] != null && prev[0].get(1).isPositive()) {
                    Result result = new Result();
                    result.prev = prev[0];
                    result.curr = vec;
                    result.pred = predictionService.makePrediction(prev[0], vec);
                    result.tick = map;
                    result.recoupledPrediction = decoupleService.recouple(symbolList, result.pred);
                    results.add(result);
                }
                prev[0] = vec;
            }
        });
        return results;
    }

    private List<ProbabilityVector> insertTicksFromStateMap(Map<SymbolPair, Real> map) {
        List<ProbabilityVector> probs = Lists.newArrayList();
        map.keySet().forEach(pair -> {
            Tick tick = Tick.builder().symbolPair(pair).val(map.get(pair).toDouble()).timestamp(Instant.now()).build();
            decoupleService.insertTick(tick);
        });
        probs.add(decoupleService.decouple());
        return probs;
    }

    @SuppressWarnings("unused")
    class Result {
        ProbabilityVector prev;
        ProbabilityVector curr;
        ProbabilityVector pred;
        Map<SymbolPair, Real> tick;
        Map<SymbolPair, Real> recoupledPrediction;

        @Override
        public String toString() {
            if (pred == null) {
                return "undef";
            }
            return pred.toString();
        }
    }

//    public ProbabilityVector getDecoupledSystemForNewTick(Map<SymbolPair, Real> map, SymbolPair pair) {
//        Tick tick = Tick.builder().symbolPair(pair).val(map.get(pair).toDouble()).timestamp(Instant.now()).build();
//        decoupleService.insertTick(tick);
//        return decoupleService.decouple();
//    }
}