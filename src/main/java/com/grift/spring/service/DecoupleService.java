package com.grift.spring.service;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import com.grift.forex.symbol.SymbolPair;
import com.grift.math.ProbabilityVector;
import com.grift.math.decoupler.DecouplerMatrix;
import com.grift.math.decoupler.Factory;
import com.grift.model.Tick;
import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import static com.google.common.base.Preconditions.checkNotNull;

@Service
public class DecoupleService {
    @NotNull
    private final DecouplerMatrix matrix;

    @Autowired
    public DecoupleService(Factory decouplerFactory) {
        matrix = decouplerFactory.make();
    }

    private static double getRatioFromProbabilityVector(@NotNull ProbabilityVector vector, SymbolPair symbolPair) {
        return vector.get(symbolPair.getFirst()) / vector.get(symbolPair.getSecond());
    }

    public void insertTick(@NotNull Tick tick) {
        matrix.put(checkNotNull(tick).getSymbolPair(), tick.getVal());
    }

    @NotNull
    public ProbabilityVector decouple() {
        return matrix.decouple();
    }

    @NotNull
    public Map<SymbolPair, Double> recouple(@NotNull List<SymbolPair> symbolPairs, @NotNull ProbabilityVector vector) {
        return symbolPairs.stream().collect(Collectors.toMap(symbolPair -> symbolPair, symbolPair -> getRatioFromProbabilityVector(vector, symbolPair), (a, b) -> b));
    }
}
