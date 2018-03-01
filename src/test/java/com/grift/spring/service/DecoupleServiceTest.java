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
import com.grift.math.decoupler.DecouplerMatrixEJMLImpl;
import com.grift.math.decoupler.Factory;
import com.grift.math.real.Real;
import com.grift.math.stats.ProbabilityVector;
import com.grift.model.Tick;
import org.jetbrains.annotations.NotNull;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringRunner;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@RunWith(SpringRunner.class)
@SpringBootTest(classes = DecoupleService.class)
@ContextConfiguration(classes = GriftApplication.class)
public class DecoupleServiceTest {

    @Deprecated //TODO: Fix accuracy!
    private DecoupleService decoupleService;

    private static Real getPairedValueFromValueMap(Map<String, Real> valMap, SymbolPair symbolPair) {
        return valMap.get(symbolPair.getFirst()).divide(valMap.get(symbolPair.getSecond()));
    }

    @Before
    public void setup() {
        SymbolIndexMap symbolIndexMap = new SymbolIndexMap().addSymbols(Lists.newArrayList("USD", "CAD", "GBP"));
        Factory factory = new DecouplerMatrixEJMLImpl.EJMLFactory(symbolIndexMap);
        decoupleService = new DecoupleService(factory);
    }

    @Test
    public void insertTick() {
        insertTick("USDCAD", new Real(10.12));
    }

    @Test
    public void insertTickUpdates() {
        Real USD = new Real(10);
        Real CAD = new Real(20);
        Real GBP = new Real(5);

        ProbabilityVector result;
        insertTick("USDCAD", USD.divide(CAD));
        insertTick("GBPCAD", GBP.divide(CAD));
        result = decoupleService.decouple();
        assertEquals(USD.divide(CAD), result.get("USD").divide(result.get("CAD")).setDigitsPrecision(2));
        USD = new Real(15);
        insertTick("USDCAD", USD.divide(CAD));
        result = decoupleService.decouple();
        assertEquals(USD.divide(CAD), result.get("USD").divide(result.get("CAD")).setDigitsPrecision(2));
    }

    private void insertTick(String pair, Real val) {
        decoupleService.insertTick(new Tick(new SymbolPair(pair), val.toDouble(), Instant.now()));
    }

    @Test
    public void testDecouple() {
        Map<String, Real> valMap = new HashMap<>();
        valMap.put("CAD", new Real(100));
        valMap.put("USD", new Real(12));
        valMap.put("GBP", new Real(700));
        valMap.put("EUR", new Real(70));

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
        Map<SymbolPair, Real> result = decoupleService.recouple(symbolPairs, decoupling);

        assertEquals("sizes differ?", symbolPairs.size(), result.size());
        symbolPairs.forEach(pair -> {
            assertTrue("missing pair", result.containsKey(pair));
            assertEquals("values differ", getPairedValueFromValueMap(valMap, pair).setDigitsPrecision(14), result.get(pair).setDigitsPrecision(14));
        });
    }

    private void insertAllTicksFromValueMap(Map<String, Real> valMap) {
        valMap.keySet().forEach(sym1 -> valMap.keySet().stream().filter(sym2 -> !sym1.equals(sym2)).forEachOrdered(sym2 -> insertTick(sym1 + sym2, getPairedValueFromValueMap(valMap, new SymbolPair(sym1, sym2)))));
    }

    @NotNull
    private Factory getFactory(@NotNull ImmutableSymbolIndexMap map) {
        return new DecouplerMatrixEJMLImpl.EJMLFactory(map);
    }
}