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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringRunner;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@RunWith(SpringRunner.class)
@SpringBootTest(classes = DecoupleService.class)
@ContextConfiguration(classes = GriftApplication.class)
public class DecoupleServiceTest {

    private static final double EPSILON = 0.000000001;
    @Autowired
    private DecoupleService decoupleService;

    @Autowired
    private Factory factory;

    @Test
    public void insertTick() {
        decoupleService.insertTick(new Tick(new SymbolPair("USDCAD"), 10.12, Instant.now()));
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
        DecouplerMatrixColtImpl.ColtFactory factory = new DecouplerMatrixColtImpl.ColtFactory(map);
        decoupleService = new DecoupleService(factory);

        for (String sym1 : valMap.keySet()) {
            for (String sym2 : valMap.keySet()) {
                if (sym1.equals(sym2)) continue;
                double val = valMap.get(sym1) / valMap.get(sym2);
                decoupleService.insertTick(new Tick(new SymbolPair(sym1 + sym2), val, Instant.now()));
            }
        }

        ProbabilityVector decoupling = decoupleService.decouple();
        Map<SymbolPair, Double> result = decoupleService.recouple(symbolPairs, decoupling);

        assertEquals("sizes differ?", symbolPairs.size(), result.size());
        for (SymbolPair pair : symbolPairs) {
            assertTrue("missing pair", result.containsKey(pair));
            assertEquals("values differ", getPairedValueFromValueMap(valMap, pair), result.get(pair), EPSILON);
        }
    }

    private double getPairedValueFromValueMap(Map<String, Double> valMap, SymbolPair symbolPair) {
        return valMap.get(symbolPair.getFirst()) / valMap.get(symbolPair.getSecond());
    }
}