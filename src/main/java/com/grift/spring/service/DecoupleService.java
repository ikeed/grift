package com.grift.spring.service;

import com.grift.math.decoupler.DecouplerMatrix;
import com.grift.math.decoupler.Factory;
import com.grift.model.Tick;
import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class DecoupleService {
    private final DecouplerMatrix matrix;

    @Autowired
    public DecoupleService(Factory decouplerFactory) {
        matrix = decouplerFactory.make();
    }

    public void insertTick(@NotNull Tick tick) {
        matrix.put(tick.getSymbolPair(), tick.getVal());
    }
}
