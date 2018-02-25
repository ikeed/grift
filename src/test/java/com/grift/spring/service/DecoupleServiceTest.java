package com.grift.spring.service;

import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import com.google.common.collect.Lists;
import com.grift.GriftApplication;
import com.grift.forex.symbol.ImmutableSymbolIndexMap;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.forex.symbol.SymbolPair;
import com.grift.math.ProbabilityVector;
import com.grift.math.decoupler.DecouplerMatrixColtImpl;
import com.grift.math.decoupler.Factory;
import com.grift.model.Tick;
import org.jetbrains.annotations.NotNull;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringRunner;

import static org.apache.commons.math.util.MathUtils.EPSILON;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@RunWith(SpringRunner.class)
@SpringBootTest(classes = DecoupleService.class)
@ContextConfiguration(classes = GriftApplication.class)
public class DecoupleServiceTest {

    private DecoupleService decoupleService;

    private static double getPairedValueFromValueMap(Map<String, Double> valMap, SymbolPair symbolPair) {
        return valMap.get(symbolPair.getFirst()) / valMap.get(symbolPair.getSecond());
    }

    @Before
    public void setup() {
        SymbolIndexMap symbolIndexMap = new SymbolIndexMap().addSymbols(Lists.newArrayList("USD", "CAD", "GBP"));
        Factory factory = new DecouplerMatrixColtImpl.ColtFactory(symbolIndexMap);
        decoupleService = new DecoupleService(factory);
    }

    @Test
    public void insertTick() {
        insertTick("USDCAD", 10.12);
    }

    @Test
    public void insertTickUpdates() {
        double USD = 10;
        double CAD = 20;
        double GBP = 5;

        ProbabilityVector result;
        insertTick("USDCAD", USD / CAD);
        insertTick("GBPCAD", GBP / CAD);
        result = decoupleService.decouple();
        assertEquals(USD / CAD, result.get("USD") / result.get("CAD"), EPSILON);
        USD = 15;
        insertTick("USDCAD", USD / CAD);
        result = decoupleService.decouple();
        assertEquals(USD / CAD, result.get("USD") / result.get("CAD"), EPSILON);
    }

    private void insertTick(String pair, double val) {
        decoupleService.insertTick(new Tick(new SymbolPair(pair), val, Instant.now()));
    }

    @Test
    public void testDecouple() {
        Map<String, Double> valMap = new HashMap<>();
        valMap.put("CAD", 100d);
        valMap.put("USD", 12d);
        valMap.put("GBP", 700d);
        valMap.put("EUR", 50d);

        final List<SymbolPair> symbolPairs = Lists.newArrayList(
                new SymbolPair("CADUSD"),
                new SymbolPair("USDGBP"),
                new SymbolPair("GBPEUR")
        );

        ImmutableSymbolIndexMap map = new SymbolIndexMap().addSymbols(Lists.newArrayList(valMap.keySet())).getImmutableCopy();
        Factory factory = getFactory(map);
        decoupleService = new DecoupleService(factory);

        insertAllTicksFromValueMap(valMap);

        ProbabilityVector decoupling = decoupleService.decouple();
        Map<SymbolPair, Double> result = decoupleService.recouple(symbolPairs, decoupling);

        assertEquals("sizes differ?", symbolPairs.size(), result.size());
        symbolPairs.forEach(pair -> {
            assertTrue("missing pair", result.containsKey(pair));
            assertEquals("values differ", getPairedValueFromValueMap(valMap, pair), result.get(pair), 1000 * EPSILON);
        });
    }

    private void insertAllTicksFromValueMap(Map<String, Double> valMap) {
        valMap.keySet().forEach(sym1 -> valMap.keySet().stream().filter(sym2 -> !sym1.equals(sym2)).forEachOrdered(sym2 -> insertTick(sym1 + sym2, getPairedValueFromValueMap(valMap, new SymbolPair(sym1, sym2)))));
    }

    @NotNull
    private Factory getFactory(@NotNull ImmutableSymbolIndexMap map) {
        return new DecouplerMatrixColtImpl.ColtFactory(map);
    }
}