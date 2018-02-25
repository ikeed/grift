package com.grift.spring.controller;

import java.util.List;
import java.util.Map;
import com.google.common.collect.Lists;
import com.grift.forex.symbol.SymbolPair;
import com.grift.math.ProbabilityVector;
import com.grift.spring.service.DecoupleService;
import com.grift.spring.service.PredictionService;
import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.context.annotation.ScopedProxyMode;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RequestMapping("/api/tick")
@RestController
@Scope(value = "request", proxyMode = ScopedProxyMode.TARGET_CLASS)
public class PredictionController {

    private final DecoupleService decoupleService;

    private final PredictionService predictionService;

    private final List<SymbolPair> symbolList;

    @Autowired
    public PredictionController(DecoupleService decoupleService, PredictionService predictionService, List<SymbolPair> symbolList) {
        this.decoupleService = decoupleService;
        this.predictionService = predictionService;
        this.symbolList = symbolList;
    }

    @NotNull
    @RequestMapping(method = RequestMethod.GET, produces = {MediaType.APPLICATION_JSON_VALUE}, path = "/prediction")
    public List<Map<SymbolPair, Double>> getPrediction(@NotNull @RequestBody List<ProbabilityVector> vectors) {
        final List<Map<SymbolPair, Double>> list = Lists.newArrayList();

        for (int i = 0; i < vectors.size() - 1; i++) {
            ProbabilityVector prediction = predictionService.makePrediction(vectors.get(i), vectors.get(i + 1));
            list.add(decoupleService.recouple(symbolList, prediction));
        }
        return list;
    }
}
