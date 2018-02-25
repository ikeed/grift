package com.grift.spring.service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import com.grift.forex.symbol.SymbolPair;
import com.grift.math.ProbabilityVector;
import com.grift.math.decoupler.DecouplerMatrix;
import com.grift.math.decoupler.Factory;
import com.grift.model.Tick;
import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class DecoupleService {
    @NotNull
    private final DecouplerMatrix matrix;

    @Autowired
    public DecoupleService(Factory decouplerFactory) {
        matrix = decouplerFactory.make();
    }

    public void insertTick(@NotNull Tick tick) {
        matrix.put(tick.getSymbolPair(), tick.getVal());
    }

    @NotNull
    public ProbabilityVector decouple() {
        return matrix.decouple();
    }

    @NotNull
    public Map<SymbolPair, Double> recouple(@NotNull List<SymbolPair> symbolPairs, @NotNull ProbabilityVector vector) {
        Map<SymbolPair, Double> map = new HashMap<>();
        for (SymbolPair pair : symbolPairs) {
            map.put(pair, vector.get(pair.getFirst()) / vector.get(pair.getSecond()));
        }
        return map;
    }
}
